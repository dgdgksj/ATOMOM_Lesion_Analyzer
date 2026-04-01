import os
import pstats
import pandas as pd

# .prof 파일이 위치한 디렉토리
directory = './qwer/'  # 이 부분을 실제 디렉토리 경로로 변경하세요.

# 디렉토리 내의 모든 .prof 파일 찾기
prof_files = [f for f in os.listdir(directory) if f.endswith('.prof')]

# 각 .prof 파일의 이름과 총 소요 시간(밀리초로 변환) 저장
data = []

for prof_file in prof_files:
    file_path = os.path.join(directory, prof_file)
    stats = pstats.Stats(file_path)
    total_time_ms = stats.total_tt * 1000  # 초를 밀리초로 변환
    data.append((prof_file, total_time_ms))

# DataFrame으로 변환
df = pd.DataFrame(data, columns=['File Name', 'Total Time (ms)'])

# 엑셀 파일로 저장
excel_file_path = 'profiling_times_ms.xlsx'  # 원하는 파일 경로와 이름으로 변경하세요.
df.to_excel(excel_file_path, index=False)

print(f"Saved profiling data to {excel_file_path}")
