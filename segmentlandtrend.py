import numpy as np
import os
import folium
import time
import pwlf
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import math
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from dask import delayed
from dask.distributed import Client, LocalCluster



"""Kalman Filter: Phù hợp với dữ liệu NDVI có nhiễu ngẫu nhiên cao hoặc chuỗi dài hạn có sự thay đổi từ từ, ví dụ khi cần phát hiện xu hướng dài hạn mà không bị ảnh hưởng quá nhiều bởi nhiễu.
Savitzky-Golay Filter: Tốt nhất khi bạn muốn giữ lại các biến động chu kỳ mà không làm biến dạng dữ liệu, đặc biệt hữu ích khi bạn muốn theo dõi các thay đổi theo mùa trong NDVI.
=> Khu vực có người: Savgol, khu vực rừng: Kalman
"""

np.random.seed(42)


class PixelMapGenerator:
    def __init__(self, all_flag):
        """
        Khởi tạo lớp với dữ liệu all_flag.
        """
        self.all_flag = all_flag

    @staticmethod
    def calculate_pixel_coords(lat, lon, size=30):
        """
        Tính tọa độ góc pixel chứa một điểm.
        """
        delta_lat = size / 111000  # Δvĩ độ (30m)
        delta_lon = size / (111000 * np.cos(np.radians(lat)))  # Δkinh độ (30m)
        # Tính tọa độ các góc của pixel
        return [
            [lat - delta_lat / 2, lon - delta_lon / 2],  # Bottom-left
            [lat - delta_lat / 2, lon + delta_lon / 2],  # Bottom-right
            [lat + delta_lat / 2, lon + delta_lon / 2],  # Top-right
            [lat + delta_lat / 2, lon - delta_lon / 2],  # Top-left
            [lat - delta_lat / 2, lon - delta_lon / 2],  # Close polygon
        ]

    def generate_map(self, file_name="fill_map_color.html"):
        """
        Tạo bản đồ từ dữ liệu all_flag và lưu vào file HTML.
        """
        # Tính trung tâm bản đồ
        map_center = [
            np.mean([pixel["latitude"] for pixel in self.all_flag.values()]),
            np.mean([pixel["longitude"] for pixel in self.all_flag.values()])
        ]
        m = folium.Map(location=map_center, zoom_start=15)

        # Thêm lớp bản đồ vệ tinh Google
        folium.TileLayer(
            tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Satellite",
            subdomains=['mt0', 'mt1', 'mt2', 'mt3'],
            max_zoom=20,
        ).add_to(m)

        # Tạo LayerGroup để chứa các lớp tô màu
        fill_layer = folium.FeatureGroup(name="Fill Layer")

        # Chuẩn hóa màu và ánh xạ
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["darkred", "red", "white", "green", "darkgreen"])

        # Duyệt qua các pixel trong all_flag
        for pixel_id, pixel_data in self.all_flag.items():
            lat = pixel_data["latitude"]
            lon = pixel_data["longitude"]
            slope = pixel_data["slope"]
            change_year = pixel_data["change_year"]

            # Ánh xạ giá trị slope sang màu
            color = mcolors.to_hex(cmap(norm(slope)))

            # Tính tọa độ pixel chứa điểm
            pixel_coords = self.calculate_pixel_coords(lat, lon)

            # Vẽ pixel vào LayerGroup
            folium.Polygon(
                locations=pixel_coords,
                color="transparent",  # Không vẽ viền
                fill=True,
                fill_color=color,  # Màu tô
                fill_opacity=0.8,
                tooltip=f"Pixel ID: {pixel_id}<br>Year: {change_year}<br>Slope: {slope:.2f}<br>Longitude: {lon}<br>Latitude: {lat}"
            ).add_to(fill_layer)

        # Thêm LayerGroup vào bản đồ
        fill_layer.add_to(m)

        # Thêm điều khiển layer để bật/tắt lớp
        folium.LayerControl(collapsed=False).add_to(m)

        # Lưu và hiển thị bản đồ
        m.save(file_name)
        print(f"Lưu bản đồ vào {file_name}")




class ChangeDetection:

    def __init__(self, params, data, min_year):
        """
        Khởi tạo lớp ChangeDetection với đầu vào từ người dùng.
        """
        # Nhập các tham số từ người dùng, với giá trị mặc định
        self.n_segments = params["n_segments"]
        self.vertex_count_overshoot = params["vertex_count_overshoot"]
        self.recovery_threshold = params["recovery_threshold"]
        self.minObservation = params["minObservation"]
        self.human_affected = params["human_affected"]

        self.data = data
        self.min_year = min_year
        self.smoothed_data = None
        self.smoothed_data_normalized = None
        self.breakpoints = None
        self.slopes = None
        self.pwlf_model = None

    @staticmethod
    def calculate_angle_between_slopes(m1, m2):
        """
        Tính góc giữa hai đoạn thẳng dựa trên độ dốc.
        """
        cos_theta = abs((1 + m1 * m2) / (math.sqrt(1 + m1**2) * math.sqrt(1 + m2**2)))
        return math.degrees(math.acos(cos_theta))


    def calculate_rmse(self):
        """
        Tính RMSE (Root Mean Square Error) của mô hình LandTrendr.
        """
        y_true = self.smoothed_data
        y_pred = self.pwlf_model.predict(np.arange(len(y_true)))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse
    
    @staticmethod
    def calculate_variance(y_true, y_pred):
        """
        Tính phương sai giữa giá trị thực và giá trị dự đoán.
        """
        return np.mean((y_true - y_pred) ** 2)

    def adjust_duplicates(self, lst):
        lst = [round(num) for num in lst]
        max_value = lst[-1]  # Giá trị lớn nhất ban đầu trong danh sách
        i = 0
        while i < len(lst) - 1:
            # Nếu phần tử hiện tại trùng với phần tử tiếp theo
            if lst[i] == lst[i + 1]:
                lst[i] += 1  # Cộng thêm 1 cho phần tử hiện tại
                
                # Kiểm tra nếu giá trị mới lớn hơn max_value, loại bỏ phần tử
                if lst[i] > max_value:
                    lst.pop(i)
                    i -= 1  # Điều chỉnh chỉ số để không bỏ qua phần tử nào
                else:
                    lst.sort()
                    i -= 1
            i += 1
        return lst


    def smooth_spikes(self):
        """
        Làm mịn dữ liệu để loại bỏ nhiễu bằng cách lấy trung bình của các điểm liền kề khi phát hiện nhiễu.
        """
        ndvi_values = self.data
        # smoothed_values = ndvi_values.copy()
        smoothed_values = savgol_filter(ndvi_values, window_length=7, polyorder=5)
        # for i in range(1, len(ndvi_values) - 1):
        #     if ndvi_values[i] != 0:
        #         spike_ratio_before = abs(ndvi_values[i] - ndvi_values[i-1]) / abs(ndvi_values[i-1])
        #         spike_ratio_after = abs(ndvi_values[i+1] - ndvi_values[i]) / abs(ndvi_values[i+1])
        #         if spike_ratio_before >= self.spike_threshold and spike_ratio_after >= self.spike_threshold:
        #             smoothed_values[i] = (ndvi_values[i-1] + ndvi_values[i+1]) / 2
        # self.smoothed_data = smoothed_values
        self.smoothed_data = smoothed_values

    def segment_with_overshoot(self):
        """
        Thực hiện hồi quy phân đoạn với số lượng đoạn tối đa được phép và thêm overshoot.
        """
        # Chuẩn hóa smoothed_data
        self.smoothed_data_normalized = self.smoothed_data  / (self.smoothed_data.max() - self.smoothed_data.min()) * (len(self.smoothed_data) - 1)
        
        # Thực hiện hồi quy phân đoạn với dữ liệu đã chuẩn hóa
        time_numeric = np.arange(len(self.smoothed_data_normalized))
        my_pwlf = pwlf.PiecewiseLinFit(time_numeric, self.smoothed_data_normalized)
        breakpoints = my_pwlf.fitfast(self.n_segments + self.vertex_count_overshoot)
        self.breakpoints = self.adjust_duplicates(breakpoints)
        my_pwlf.fit_with_breaks(self.breakpoints)
        self.slopes = my_pwlf.slopes
        self.pwlf_model = my_pwlf

    def dynamic_thresholds(self, slopes, ndvi_values, ndvi_percentile):
        """
        Tính toán các ngưỡng động dựa trên độ dốc và sự thay đổi NBR.
        """
        result = {}
        for i in range(len(slopes) - 1):
            theta = self.calculate_angle_between_slopes(slopes[i], slopes[i+1])
            result[i] = {'slope': slopes[i], 'slope_next': slopes[i+1], 'theta': theta}
        
        ndvi_diff = np.abs(np.diff(ndvi_values))
        ndvi_threshold = np.percentile(ndvi_diff, ndvi_percentile)
        
        return result, ndvi_threshold
    
    def filter_breakpoints_dynamic(self, breakpoints, slopes, ndvi_values):
        """
        Lọc các điểm phân đoạn dựa trên ngưỡng động.
        """

        # Tính ndvi_percentile
        ndvi_percentile = (50 / (1 + 0.1 * (len(ndvi_values)-1) * np.std(ndvi_values))) * (1 + (self.n_segments ** 2))
        ndvi_percentile = min(ndvi_percentile, 50)  # Giới hạn trên

        slope_threshold_result, ndvi_threshold = self.dynamic_thresholds(slopes, ndvi_values, ndvi_percentile)
        
        filtered_breakpoints = [breakpoints[0]]  # Giữ lại điểm đầu tiên
        
        for i in range(1, len(breakpoints) - 1):
            delta_nbr_after = abs(ndvi_values[breakpoints[i]+1] - self.pwlf_model.predict([breakpoints[i]])[0])
            delta_nbr_before = abs(ndvi_values[breakpoints[i]-1] - self.pwlf_model.predict([breakpoints[i]])[0])
            
            # Kiểm tra các điều kiện dựa trên ngưỡng động
            if (delta_nbr_after > ndvi_threshold or delta_nbr_before > ndvi_threshold) and \
               (((slope_threshold_result[i-1]['slope'] * slope_threshold_result[i-1]['slope_next']) < 0) or \
                ((slope_threshold_result[i-1]['slope'] * slope_threshold_result[i-1]['slope_next']) >= 0 and \
                 slope_threshold_result[i-1]['theta'] >= 30)):
                filtered_breakpoints.append(breakpoints[i])
        
        # Giữ lại điểm cuối cùng
        filtered_breakpoints.append(breakpoints[-1])
        
        self.breakpoints = filtered_breakpoints

    def remove_breakpoint_with_max_variance(self, time_numeric, ndvi_values, max_segments):
        """
        Loại bỏ đỉnh gây ra sự gia tăng phương sai lớn nhất bằng cách sử dụng pwlf.
        """
        iteration = 1  # Biến để đếm số lần lặp
        while len(self.breakpoints) > max_segments + 1:  # Cộng 1 vì giữ điểm đầu và cuối
            variances = []
            temp_pwlf = []
            # Duyệt qua tất cả các đỉnh, bỏ từng đỉnh một và tính phương sai
            for i in range(1, len(self.breakpoints) - 1):  # Bỏ qua đỉnh đầu và cuối
                # Tạo danh sách breakpoints sau khi loại bỏ đỉnh thứ i
                temp_breakpoints = self.breakpoints[:i] + self.breakpoints[i+1:]
                
                # Sử dụng pwlf để thực hiện hồi quy phân đoạn với breakpoints tạm thời
                my_pwlf = pwlf.PiecewiseLinFit(time_numeric, ndvi_values)
                my_pwlf.fit_with_breaks(temp_breakpoints)
                
                temp_pwlf.append(my_pwlf)
                # Dự đoán giá trị NBR sau khi hồi quy
                y_pred = my_pwlf.predict(time_numeric)

                # Tính phương sai giữa giá trị thực và giá trị dự đoán
                total_variance = self.calculate_variance(ndvi_values, y_pred)
                
                # Lưu lại phương sai khi loại đỉnh thứ i
                variances.append(total_variance)
            
            # Tìm đỉnh có phương sai lớn nhất và loại bỏ
            max_variance_idx = np.argmin(variances) + 1  # Cộng 1 vì bỏ qua đỉnh đầu
            self.pwlf_model = temp_pwlf[np.argmin(variances)]
            # Loại bỏ đỉnh này khỏi danh sách breakpoints
            del self.breakpoints[max_variance_idx]
            # Tăng biến đếm lần lặp
            iteration += 1
        self.slopes = self.pwlf_model.slopes

    def detect_and_remove_short_term_recovery(self):
        """
        Loại bỏ các phục hồi ngắn hạn (1 năm) trong danh sách breakpoints.
        """
        filtered_breakpoints = [self.breakpoints[0]]  # Giữ lại điểm đầu tiên
        
        for i in range(1, len(self.breakpoints) - 1):
            # Tính thời gian giữa các điểm breakpoints
            duration = self.breakpoints[i + 1] - self.breakpoints[i]
            recovery_speed = (self.pwlf_model.predict(self.breakpoints[i + 1]) - self.pwlf_model.predict(self.breakpoints[i])) / \
                             (self.pwlf_model.predict(self.breakpoints[i - 1]) - self.pwlf_model.predict(self.breakpoints[i]))
            
            # Phát hiện phục hồi ngắn hạn
            if (self.slopes[i - 1] < 0 and self.slopes[i] > 0) and (0.6 <= recovery_speed <= 1.4) and duration <= 1:
                continue  # Bỏ qua điểm phục hồi ngắn hạn này
            
            filtered_breakpoints.append(self.breakpoints[i])
        
        # Giữ lại điểm cuối cùng
        filtered_breakpoints.append(self.breakpoints[-1])
        
        self.breakpoints = filtered_breakpoints  # Cập nhật breakpoints đã loại bỏ phục hồi ngắn hạn

    def detect_and_remove_small_recoveries(self):
        """
        Loại bỏ các phục hồi không đáng kể trong danh sách breakpoints.
        
        Parameters:
        - recovery_threshold: Ngưỡng phục hồi tối thiểu để được coi là đáng kể.
        """
        recovery_threshold = self.recovery_threshold

        filtered_breakpoints = [self.breakpoints[0]]  # Giữ lại điểm đầu tiên
        
        for i in range(1, len(self.breakpoints) - 1):
            if self.slopes[i - 1] < 0 and self.slopes[i] > 0:  # Nếu đây là phục hồi
                duration = self.breakpoints[i + 1] - self.breakpoints[i]
                recovery_magnitude = ((self.pwlf_model.predict(self.breakpoints[i + 1])[0] - self.pwlf_model.predict(self.breakpoints[i])[0]) / duration) / \
                                     (self.pwlf_model.predict(self.breakpoints[i - 1])[0] - self.pwlf_model.predict(self.breakpoints[i])[0])
                
                if not (duration >= 2 and recovery_magnitude > recovery_threshold):
                    filtered_breakpoints.append(self.breakpoints[i])
            else:
                filtered_breakpoints.append(self.breakpoints[i])
        
        # Giữ lại điểm cuối cùng
        filtered_breakpoints.append(self.breakpoints[-1])
        
        self.breakpoints = filtered_breakpoints
    
    def final_model(self):
        time_numeric = np.arange(len(self.smoothed_data))
        my_pwlf = pwlf.PiecewiseLinFit(time_numeric, self.smoothed_data)
        my_pwlf.fit_with_breaks(self.breakpoints)
        self.pwlf_model = my_pwlf
        self.slopes = my_pwlf.slopes

    def plot_result(self, pixel_id):
        """
        Vẽ đồ thị kết quả cuối cùng với các breakpoints đã lọc.
        """
        time_numeric = np.arange(len(self.smoothed_data))
        plt.figure(figsize=(12, 6))
        plt.plot(time_numeric + self.min_year, self.smoothed_data, 'o-', label='Smoothed Data')
        plt.plot(time_numeric + self.min_year, self.data, 'v-', label='Origin Data')
        y_pred = self.pwlf_model.predict(time_numeric)
        rmse = np.sqrt(self.calculate_variance(self.smoothed_data, y_pred))
        plt.plot([], [], ' ', label=f'RMSE: {rmse:.4f}')
                
        for i in range(1, len(self.breakpoints)):
            x_segment = np.linspace(self.breakpoints[i-1], self.breakpoints[i], num=100)
            y_segment = self.pwlf_model.predict(x_segment)
            plt.plot(x_segment + self.min_year, y_segment, '-', color='red', linewidth = 3)
        
        for bp in self.breakpoints:
            plt.axvline(x=bp + self.min_year, linestyle='--', color='green', label=f'Breakpoint at {(bp + self.min_year):.2f}')
        
        plt.title(f"Final Piecewise Linear Regression with Filtered Breakpoints For Pixel {pixel_id}")
        plt.xlabel('Time')
        plt.ylabel('NDVI')
        plt.ylim(-1, 1)
        plt.legend()
        plt.grid(True)
        output_dir = "image"
        os.makedirs(output_dir, exist_ok=True)

        # Lưu hình ảnh vào thư mục "image"
        plt.savefig(os.path.join(output_dir, f"pixel_{pixel_id}.png"))
        plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ


@delayed
def process_pixel_dask(pixel_id, params, data, min_year, longtitude, latitude):
    # Khởi tạo một đối tượng ChangeDetection mới cho mỗi pixel
    detector = ChangeDetection(params, data, min_year)
    # Kiểm tra nếu dữ liệu không đủ số lượng quan sát
    if detector.data is not None and len(detector.data) < detector.minObservation:
        raise ValueError(f"Dữ liệu không đủ số lượng quan sát tối thiểu: yêu cầu {detector.minObservation}, nhưng chỉ có {len(detector.data)}.")
    # Tiến hành các bước phân tích
    detector.smooth_spikes()
    detector.segment_with_overshoot()
    detector.filter_breakpoints_dynamic(detector.breakpoints, detector.slopes, detector.smoothed_data_normalized)
    detector.remove_breakpoint_with_max_variance(np.arange(len(detector.smoothed_data_normalized)), detector.smoothed_data_normalized, detector.n_segments)
    
    if detector.human_affected == False:
        detector.detect_and_remove_short_term_recovery()
        detector.detect_and_remove_small_recoveries()

    detector.final_model()
    detector.plot_result(pixel_id)
    return {
        "pixel_id": pixel_id,
        "slope": detector.slopes[-1],
        "longitude": longtitude,
        "latitude": latitude,
        "change_year": detector.breakpoints[-2] + detector.min_year
    }


def run_parallel_with_client(params, all_data, min_year):
    # Tạo LocalCluster
    cluster = LocalCluster(n_workers=6, threads_per_worker=2, memory_limit='2GB')

    # Kết nối Client tới cluster
    client = Client(cluster)

    # Kiểm tra trạng thái của cluster
    print(client)
    print("Link dashboard: ", cluster.dashboard_link)


    print("\nDanh sách các nút trong cluster:")
    # Truy cập thông tin scheduler
    print(f"Scheduler: {cluster.scheduler_info['address']}")  # Truy cập trực tiếp vào 'address'
    print("Danh sách Worker: ")
    for worker in cluster.workers.values():
        print(f"Worker ID: {worker.id}, Address: {worker.address}")


    # Tạo danh sách các tác vụ delayed cho tất cả các pixel
    tasks = [
        process_pixel_dask(
            pixel_id.item(),
            params,
            all_data.NDVI.sel(pixel_id=pixel_id).values,
            min_year,
            all_data.longitude.sel(pixel_id=pixel_id).item(),
            all_data.latitude.sel(pixel_id=pixel_id).item()
        )
        for pixel_id in all_data.pixel_id
    ]

    # Chạy song song và theo dõi tiến trình
    start_time = time.time()
    futures = client.compute(tasks)
    results = [future.result() for future in futures]
    all_flag = {result["pixel_id"]: result for result in results}
    end_time = time.time()
    client.close()  # Đóng client khi xong

    print("Thành công!")
    print(f"Thời gian chạy: {end_time - start_time:.2f} giây")

    return all_flag

def import_data(file_path):
    # Đọc tệp CSV vào DataFrame
    data = pd.read_csv(file_path) 
    data.replace(["", "NA", "null", "-"], np.nan, inplace=True)
    
    # Trích xuất danh sách năm, pixel_id và tọa độ
    years = [int(col.split('_')[1]) for col in data.columns if col.startswith("Year_")]
    pixel_ids = data['pixel_id']
    
    # Tạo xarray.Dataset và thêm longitude, latitude vào như là tọa độ
    ds = xr.Dataset(
        {
            "NDVI": (["pixel_id", "time"], data.drop(columns=["pixel_id", "longitude", "latitude", "index"]).values)
        },
        coords={
            "pixel_id": pixel_ids,
            "time": years,
            "longitude": ("pixel_id", data["longitude"].values),  # Thêm tọa độ longitude
            "latitude": ("pixel_id", data["latitude"].values)    # Thêm tọa độ latitude
        }
    )

    # Điền giá trị NaN
    ds_filled = ds.interpolate_na(dim="time", method="linear")

    return ds_filled, min(years)

params = {
    "n_segments": 4,
    # "spike_threshold": 0.7,
    "vertex_count_overshoot": 3,
    "recovery_threshold": 0.25,
    "minObservation": 6,
    "human_affected": False
}

if __name__ == '__main__':
    all_data, min_year = import_data('pivot_data.csv')
    all_flag = run_parallel_with_client(params, all_data, min_year)
    map = PixelMapGenerator(all_flag)
    map.generate_map()