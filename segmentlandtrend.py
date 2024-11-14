import numpy as np
import pwlf
import matplotlib.pyplot as plt
import math
import pandas as pd
import xarray as xr
import time

np.random.seed(42)

class ChangeDetection:

    def __init__(self):
        """
        Khởi tạo lớp ChangeDetection với đầu vào từ người dùng.
        """
        self.data = None

        # Nhập các tham số từ người dùng, với giá trị mặc định
        self.n_segments = int(input("Nhập số đoạn phân đoạn tối đa (mặc định là 4): ") or 4)
        self.spike_threshold = float(input("Nhập ngưỡng phát hiện nhiễu (mặc định là 0.9): ") or 0.9)
        self.vertex_count_overshoot = int(input("Nhập số điểm gãy khúc tăng thêm cho bộ lọc (mặc định là 2): ") or 2)
        self.recovery_threshold = float(input("Nhập ngưỡng phục hồi gới hạn trên (mặc định là 0.25): ") or 0.25)
        self.dsnr = input("Chuẩn hóa theo RMSE (true/false, mặc định là false): ").strip().lower() == "true"
        self.preventOneYearRecovery = input("Ngăn chặn phục hồi ngắn hạn trong 1 năm ? (true/false, mặc định là false): ").strip().lower() == "true"
        self.minObservation = int(input("Nhập số phần tử cần thiết tối thiểu của chuỗi (mặc định là 6): ") or 6)

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

    def import_data(self, file_path, drop_columns=None):

        # Đọc tệp CSV vào DataFrame
        data = pd.read_csv(file_path)
        
        # Loại bỏ các giá trị bị thiếu
        data = data.dropna()

        # Loại bỏ các cột không cần thiết
        if drop_columns:
            data = data.drop(columns=drop_columns, errors='ignore')
        data.reset_index(drop=True, inplace=True)     

        return data

    def smooth_spikes(self):
        """
        Làm mịn dữ liệu để loại bỏ nhiễu bằng cách lấy trung bình của các điểm liền kề khi phát hiện nhiễu.
        """
        nbr_values = self.data['NBR'].values
        smoothed_values = nbr_values.copy()
        for i in range(1, len(nbr_values) - 1):
            if nbr_values[i] != 0:
                spike_ratio_before = abs(nbr_values[i] - nbr_values[i-1]) / abs(nbr_values[i-1])
                spike_ratio_after = abs(nbr_values[i+1] - nbr_values[i]) / abs(nbr_values[i+1])
                if spike_ratio_before >= self.spike_threshold and spike_ratio_after >= self.spike_threshold:
                    smoothed_values[i] = (nbr_values[i-1] + nbr_values[i+1]) / 2
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
        breakpoints = my_pwlf.fit(self.n_segments + self.vertex_count_overshoot)
        self.breakpoints = self.adjust_duplicates(breakpoints)
        my_pwlf.fit_with_breaks(self.breakpoints)
        self.slopes = my_pwlf.slopes
        self.pwlf_model = my_pwlf

    def dynamic_thresholds(self, slopes, nbr_values, nbr_percentile):
        """
        Tính toán các ngưỡng động dựa trên độ dốc và sự thay đổi NBR.
        """
        result = {}
        for i in range(len(slopes) - 1):
            theta = self.calculate_angle_between_slopes(slopes[i], slopes[i+1])
            result[i] = {'slope': slopes[i], 'slope_next': slopes[i+1], 'theta': theta}
        
        nbr_diff = np.abs(np.diff(nbr_values))
        nbr_threshold = np.percentile(nbr_diff, nbr_percentile)
        
        return result, nbr_threshold
    
    def filter_breakpoints_dynamic(self, breakpoints, slopes, nbr_values, nbr_percentile=25):
        """
        Lọc các điểm phân đoạn dựa trên ngưỡng động.
        """
        slope_threshold_result, nbr_threshold = self.dynamic_thresholds(slopes, nbr_values, nbr_percentile)
        
        filtered_breakpoints = [breakpoints[0]]  # Giữ lại điểm đầu tiên
        
        for i in range(1, len(breakpoints) - 1):
            delta_nbr_after = abs(nbr_values[breakpoints[i]+1] - self.pwlf_model.predict([breakpoints[i]])[0])
            delta_nbr_before = abs(nbr_values[breakpoints[i]-1] - self.pwlf_model.predict([breakpoints[i]])[0])
            
            # Kiểm tra các điều kiện dựa trên ngưỡng động
            if (delta_nbr_after > nbr_threshold or delta_nbr_before > nbr_threshold) and \
               (((slope_threshold_result[i-1]['slope'] * slope_threshold_result[i-1]['slope_next']) < 0) or \
                ((slope_threshold_result[i-1]['slope'] * slope_threshold_result[i-1]['slope_next']) >= 0 and \
                 slope_threshold_result[i-1]['theta'] >= 30)):
                filtered_breakpoints.append(breakpoints[i])
        
        # Giữ lại điểm cuối cùng
        filtered_breakpoints.append(breakpoints[-1])
        
        self.breakpoints = filtered_breakpoints

    def remove_breakpoint_with_max_variance(self, time_numeric, nbr_values, max_segments):
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
                my_pwlf = pwlf.PiecewiseLinFit(time_numeric, nbr_values)
                my_pwlf.fit_with_breaks(temp_breakpoints)
                
                temp_pwlf.append(my_pwlf)
                # Dự đoán giá trị NBR sau khi hồi quy
                y_pred = my_pwlf.predict(time_numeric)

                # Tính phương sai giữa giá trị thực và giá trị dự đoán
                total_variance = self.calculate_variance(nbr_values, y_pred)
                
                # Lưu lại phương sai khi loại đỉnh thứ i
                variances.append(total_variance)
            
            # Tìm đỉnh có phương sai lớn nhất và loại bỏ
            max_variance_idx = np.argmin(variances) + 1  # Cộng 1 vì bỏ qua đỉnh đầu
            self.pwlf_model = temp_pwlf[np.argmin(variances)]
            # Loại bỏ đỉnh này khỏi danh sách breakpoints
            del self.breakpoints[max_variance_idx]
            # Tăng biến đếm lần lặp
            iteration += 1
        self.slopes = (self.pwlf_model).slopes

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
            if (self.slopes[i - 1] < 0 and self.slopes[i] > 0) and (0.9 <= recovery_speed <= 1.2) and duration <= 1:
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

    def plot_result(self):
        """
        Vẽ đồ thị kết quả cuối cùng với các breakpoints đã lọc.
        """
        time_numeric = np.arange(len(self.smoothed_data))
        plt.figure(figsize=(12, 6))
        plt.plot(time_numeric, self.smoothed_data, 'o', label='Smoothed Data')
        plt.plot(time_numeric, self.data, 'v', label='Origin Data')
        
        my_pwlf = pwlf.PiecewiseLinFit(time_numeric, self.smoothed_data)
        my_pwlf.fit_with_breaks(self.breakpoints)
                
        for i in range(1, len(self.breakpoints)):
            x_segment = np.linspace(self.breakpoints[i-1], self.breakpoints[i], num=100)
            y_segment = my_pwlf.predict(x_segment)
            plt.plot(x_segment, y_segment, '-', color='red')
        
        for bp in self.breakpoints:
            plt.axvline(x=bp, linestyle='--', color='green', label=f'Breakpoint at {bp:.2f}')
        
        plt.title('Final Piecewise Linear Regression with Filtered Breakpoints')
        plt.xlabel('Time')
        plt.ylabel('NBR')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Chạy toàn bộ quy trình phân tích.
        """
        print("Nhập dữ liệu, resize dữ liệu.")
        self.data = self.import_data('Pixel_TimeSeries_Export.csv', 'date')
        print("Bắt đầu làm mịn dữ liệu...")
        self.smooth_spikes()
        
        print("Phân đoạn dữ liệu với overshoot...")
        self.segment_with_overshoot()
        
        print("Lọc các điểm breakpoints...")
        self.filter_breakpoints_dynamic(self.breakpoints, self.slopes, self.smoothed_data_normalized)
        self.remove_breakpoint_with_max_variance(np.arange(len(self.smoothed_data_normalized)), self.smoothed_data_normalized, self.n_segments)
        self.detect_and_remove_short_term_recovery()
        self.detect_and_remove_small_recoveries()
        
        print("Vẽ kết quả cuối cùng...")
        # self.plot_result()
        # print("Quá trình hoàn tất.")

detector = ChangeDetection()

# Bắt đầu đo thời gian
start_time = time.time()

# Gọi phương thức run()
detector.run()

# Kết thúc đo thời gian
end_time = time.time()

# Tính toán thời gian chạy
execution_time = end_time - start_time
print(f"Thời gian chạy của phương thức run(): {execution_time:.4f} giây")