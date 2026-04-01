import tensorboard as tb
import tensorboard.program

# TensorBoard 인스턴스 생성
tb_program = tensorboard.program.TensorBoard()

# 로그 디렉토리 설정 (여기서는 'path/to/log-directory'를 로그 파일이 있는 경로로 변경해야 함)
tb_program.configure(argv=[None, '--logdir', 'C:/Users/user/Downloads/download/models/yolo/n_640_size_154_epoch_443'])

# TensorBoard 서버 시작
url = tb_program.launch()

print(f"TensorBoard is running at {url}")
