from flask import Flask, render_template, Response, jsonify
import cv2
from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from camera_ready import detect
from flask_socketio import SocketIO
import json
from flask_socketio import emit
import time
app = Flask(__name__)
socketio = SocketIO(app)
# user,pwd,ip="?","?"."?"
class VideoCamera(object):
    def __init__(self):
        self.count = 0
        self.video = cv2.VideoCapture("test.mp4")
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels%d" % (user, pwd, ip, 1))

        self.weights, imgsz = 'weight.pt', 640
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

    def __del__(self):
        self.video.release()

    def get_frame(self):
        for i in range(50):
            success, image = self.video.read()
        image = detect(source=image, half=self.half, model=self.model, device=self.device, imgsz=self.imgsz,
                       stride=self.stride)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


json_data = read_json_file('json/output.json')


@app.route('/')
def index():
    return render_template('index.html', data=json_data, title="智慧城管")

@app.route('/get_json_data', methods=['POST'])
def get_json_data():
    json_data = read_json_file('json/output.json')
    return json.dumps({'status':1, 'msg': '', 'data': json_data})

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def get_updated_json_data():
    while True:
        # 这里可以替换为你获取实时JSON数据的方法
        new_data = read_json_file('json/output.json')

        # 发送更新的JSON数据到页面
        emit('json_update', json.dumps(new_data), broadcast=True)
        time.sleep(5)  # 每5秒更新一次数据

# 在Flask应用启动时，开启一个新的线程获取并发送最新的JSON数据
@socketio.on('connect')
def connect():
    print('Client connected')
    socketio.start_background_task(target=get_updated_json_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)