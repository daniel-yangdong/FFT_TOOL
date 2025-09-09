import sys
import numpy as np
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QMessageBox, QGridLayout, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt

# 检查matplotlib是否可用
try:
    import matplotlib

    matplotlib.use('Qt5Agg')  # 必须在导入pyplot之前设置后端
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Matplotlib not available")

# 检查scipy是否可用
try:
    import scipy.fft as fft

    scipy_available = True
except ImportError:
    scipy_available = False
    # 使用numpy的FFT作为备选
    import numpy.fft as fft

    print("Using numpy FFT instead of scipy")


class SignalGeneratorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.signal_data = None
        self.time_axis = None

    def initUI(self):
        layout = QVBoxLayout()

        # 参数设置区域
        param_group = QGroupBox("信号参数设置")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("频率1 (Hz):"), 0, 0)
        self.freq1_input = QDoubleSpinBox()
        self.freq1_input.setRange(1, 1000)
        self.freq1_input.setValue(5)
        param_layout.addWidget(self.freq1_input, 0, 1)

        param_layout.addWidget(QLabel("振幅1:"), 0, 2)
        self.amp1_input = QDoubleSpinBox()
        self.amp1_input.setRange(0.1, 10.0)
        self.amp1_input.setValue(1.0)
        param_layout.addWidget(self.amp1_input, 0, 3)

        param_layout.addWidget(QLabel("频率2 (Hz):"), 1, 0)
        self.freq2_input = QDoubleSpinBox()
        self.freq2_input.setRange(1, 1000)
        self.freq2_input.setValue(15)
        param_layout.addWidget(self.freq2_input, 1, 1)

        param_layout.addWidget(QLabel("振幅2:"), 1, 2)
        self.amp2_input = QDoubleSpinBox()
        self.amp2_input.setRange(0.1, 10.0)
        self.amp2_input.setValue(0.5)
        param_layout.addWidget(self.amp2_input, 1, 3)

        param_layout.addWidget(QLabel("频率3 (Hz):"), 2, 0)
        self.freq3_input = QDoubleSpinBox()
        self.freq3_input.setRange(1, 1000)
        self.freq3_input.setValue(30)
        param_layout.addWidget(self.freq3_input, 2, 1)

        param_layout.addWidget(QLabel("振幅3:"), 2, 2)
        self.amp3_input = QDoubleSpinBox()
        self.amp3_input.setRange(0.1, 10.0)
        self.amp3_input.setValue(0.3)
        param_layout.addWidget(self.amp3_input, 2, 3)

        param_layout.addWidget(QLabel("采样率 (Hz):"), 3, 0)
        self.sample_rate_input = QSpinBox()
        self.sample_rate_input.setRange(100, 100000)
        self.sample_rate_input.setValue(1000)
        param_layout.addWidget(self.sample_rate_input, 3, 1)

        param_layout.addWidget(QLabel("持续时间 (秒):"), 3, 2)
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(0.1, 10.0)
        self.duration_input.setValue(1.0)
        param_layout.addWidget(self.duration_input, 3, 3)

        param_layout.addWidget(QLabel("噪声水平:"), 4, 0)
        self.noise_input = QDoubleSpinBox()
        self.noise_input.setRange(0.0, 1.0)
        self.noise_input.setValue(0.05)
        self.noise_input.setSingleStep(0.01)
        param_layout.addWidget(self.noise_input, 4, 1)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.generate_btn = QPushButton("生成信号")
        self.generate_btn.clicked.connect(self.generate_signal)
        button_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("保存信号")
        self.save_btn.clicked.connect(self.save_signal)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        # 图表区域
        if matplotlib_available:
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            self.canvas = QLabel("matplotlib不可用，无法显示图表")
            layout.addWidget(self.canvas)

        self.setLayout(layout)

    def generate_signal(self):
        # 获取参数
        freq1 = self.freq1_input.value()
        amp1 = self.amp1_input.value()
        freq2 = self.freq2_input.value()
        amp2 = self.amp2_input.value()
        freq3 = self.freq3_input.value()
        amp3 = self.amp3_input.value()
        sample_rate = self.sample_rate_input.value()
        duration = self.duration_input.value()
        noise_level = self.noise_input.value()

        # 生成时间轴
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        self.time_axis = t

        # 生成三个正弦波
        signal1 = amp1 * np.sin(2 * np.pi * freq1 * t)
        signal2 = amp2 * np.sin(2 * np.pi * freq2 * t)
        signal3 = amp3 * np.sin(2 * np.pi * freq3 * t)

        # 组合信号并添加噪声
        combined_signal = signal1 + signal2 + signal3
        noise = np.random.normal(0, noise_level, combined_signal.shape)
        self.signal_data = combined_signal + noise

        # 绘制信号图
        if matplotlib_available:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(t, self.signal_data)
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('振幅')
            ax.set_title('合成信号 (时域)')
            ax.grid(True)
            self.canvas.draw()
        else:
            self.canvas.setText(f"信号已生成: {len(self.signal_data)}个采样点")

        self.save_btn.setEnabled(True)

    def save_signal(self):
        if self.signal_data is None or self.time_axis is None:
            QMessageBox.warning(self, "警告", "没有可保存的信号数据")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存信号数据", "", "CSV文件 (*.csv);;所有文件 (*)")

        if file_path:
            try:
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['时间', '振幅'])
                    for t, value in zip(self.time_axis, self.signal_data):
                        writer.writerow([t, value])
                QMessageBox.information(self, "成功", "信号数据已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存文件时出错: {str(e)}")


class FFTAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.signal_data = None
        self.sample_rate = None

    def initUI(self):
        layout = QVBoxLayout()

        # 文件选择区域
        file_group = QGroupBox("信号文件")
        file_layout = QHBoxLayout()

        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        file_layout.addWidget(self.file_path_input)

        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)

        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 分析按钮
        self.analyze_btn = QPushButton("进行FFT分析")
        self.analyze_btn.clicked.connect(self.analyze_signal)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        # 结果显示
        result_group = QGroupBox("分析结果")
        result_layout = QGridLayout()

        result_layout.addWidget(QLabel("信噪比 (SNR):"), 0, 0)
        self.snr_label = QLabel("N/A")
        result_layout.addWidget(self.snr_label, 0, 1)

        result_layout.addWidget(QLabel("总谐波失真 (THD):"), 1, 0)
        self.thd_label = QLabel("N/A")
        result_layout.addWidget(self.thd_label, 1, 1)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 图表区域
        if matplotlib_available:
            self.figure = Figure(figsize=(10, 8))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            self.canvas = QLabel("matplotlib不可用，无法显示图表")
            layout.addWidget(self.canvas)

        self.setLayout(layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开信号数据", "", "CSV文件 (*.csv);;所有文件 (*)")
        if file_path:
            self.file_path_input.setText(file_path)

    def load_file(self):
        file_path = self.file_path_input.text()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择文件")
            return

        try:
            time_data = []
            signal_data = []
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 2:
                        time_data.append(float(row[0]))
                        signal_data.append(float(row[1]))

            if len(time_data) < 2:
                QMessageBox.warning(self, "警告", "文件中的数据不足")
                return

            self.signal_data = np.array(signal_data)
            self.time_axis = np.array(time_data)

            # 计算采样率
            time_diff = np.diff(self.time_axis)
            self.sample_rate = 1.0 / np.mean(time_diff)

            # 绘制原始信号
            if matplotlib_available:
                self.figure.clear()
                ax = self.figure.add_subplot(211)
                ax.plot(self.time_axis, self.signal_data)
                ax.set_xlabel('时间 (秒)')
                ax.set_ylabel('振幅')
                ax.set_title('原始信号 (时域)')
                ax.grid(True)
                self.canvas.draw()
            else:
                self.canvas.setText(f"信号已加载: {len(self.signal_data)}个采样点，采样率: {self.sample_rate:.2f} Hz")

            self.analyze_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件时出错: {str(e)}")

    def analyze_signal(self):
        if self.signal_data is None or self.sample_rate is None:
            QMessageBox.warning(self, "警告", "没有可分析的信号数据")
            return

        try:
            # 执行FFT
            n = len(self.signal_data)
            fft_result = fft.fft(self.signal_data)
            fft_freq = fft.fftfreq(n, 1.0 / self.sample_rate)

            # 只取正频率部分
            positive_freq_mask = fft_freq >= 0
            positive_freq = fft_freq[positive_freq_mask]
            positive_fft = np.abs(fft_result[positive_freq_mask]) / n
            positive_fft[1:] *= 2  # 除DC分量外，幅度加倍

            # 计算信噪比(SNR)
            signal_power = np.sum(positive_fft ** 2)
            noise_mask = positive_fft < 0.1 * np.max(positive_fft)
            noise_power = np.sum(positive_fft[noise_mask] ** 2)

            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = float('inf')

            # 计算总谐波失真(THD)
            fundamental_idx = np.argmax(positive_fft)
            fundamental_freq = positive_freq[fundamental_idx]
            fundamental_amp = positive_fft[fundamental_idx]

            # 找到谐波频率
            harmonic_freqs = []
            harmonic_amplitudes = []

            for i in range(2, 10):  # 检查2次到9次谐波
                harmonic_freq = i * fundamental_freq
                if harmonic_freq > positive_freq[-1]:
                    break

                # 在谐波频率附近找最大值
                freq_range = 5  # Hz
                mask = (positive_freq >= harmonic_freq - freq_range) & (positive_freq <= harmonic_freq + freq_range)
                if np.any(mask):
                    harmonic_idx = np.argmax(positive_fft[mask])
                    harmonic_amp = positive_fft[mask][harmonic_idx]
                    harmonic_freqs.append(harmonic_freq)
                    harmonic_amplitudes.append(harmonic_amp)

            # 计算THD
            harmonic_power = np.sum(np.array(harmonic_amplitudes) ** 2)
            if fundamental_amp > 0:
                thd = np.sqrt(harmonic_power) / fundamental_amp * 100
            else:
                thd = 0

            # 更新结果显示
            self.snr_label.setText(f"{snr:.2f} dB" if snr != float('inf') else "无限")
            self.thd_label.setText(f"{thd:.2f}%")

            # 绘制频谱图
            if matplotlib_available:
                ax = self.figure.add_subplot(212)
                ax.plot(positive_freq, positive_fft)
                ax.set_xlabel('频率 (Hz)')
                ax.set_ylabel('幅度')
                ax.set_title('频谱图')
                ax.grid(True)
                ax.set_xlim(0, min(200, positive_freq[-1]))  # 限制频率范围以便查看

                # 标记基频和谐波
                ax.axvline(x=fundamental_freq, color='r', linestyle='--', alpha=0.7, label='基频')
                for h_freq in harmonic_freqs:
                    ax.axvline(x=h_freq, color='g', linestyle='--', alpha=0.7, label='谐波')

                ax.legend()
                self.figure.tight_layout()
                self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析信号时出错: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('FFT信号分析工具')
        self.setGeometry(100, 100, 1000, 800)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 添加信号生成标签页
        self.signal_generator = SignalGeneratorWidget()
        self.tab_widget.addTab(self.signal_generator, "信号生成")

        # 添加FFT分析标签页
        self.fft_analysis = FFTAnalysisWidget()
        self.tab_widget.addTab(self.fft_analysis, "频谱分析")

        self.setCentralWidget(self.tab_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
