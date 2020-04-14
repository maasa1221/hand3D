# from flask import Flask
# app = Flask(__name__)


# @app.route('/')
# def hello():
#     name = "Hello World"
#     return name


# @app.route('/good')
# def good():
#     name = "Good"
#     return name


# if __name__ == "__main__":
#     app.run(debug=True)
import eval_script
import hoge
# from flask import Flask, render_template, request
import cv2                        
# app = Flask(__name__)


def image_capture():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mov', fourcc, 20.0, (640, 480))

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# @app.route("/")
# def index():
#     return render_template('index.html')


# @app.route("/test", methods=['GET', 'POST'])
# def test():
#     if request.method == 'GET':
#         res = request.args.get('get_value')
#         image_capture()
#         let = hoge.goodbye()

#     elif request.method == 'POST':
#         res = request.form['post_value']
#         let = hoge.goodbye()

#     return res


# if __name__ == "__main__":
#     app.run()

from flask import Flask, render_template, Response, request
import os
from camera import VideoCamera
import gspread
from PIL import Image
import io


app = Flask(__name__)

result = [0, 0]
flag = 0
roc_frame_count=0
img=[]
frame_list=[]

def gen2(camera):
    while True:
        frame = camera.get_frame2()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def resultf(num):
    return render_template('index.html', result=num)


def spread(result_list):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        'hand3d-272022-21ee65463d02.json', scope)
    gc = gspread.authorize(credentials)
    wks = gc.open('hand3D_datas').sheet1

    wks.append_row([str(result_list[0]), str(result_list[1])])


def test():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640,340))
    # gen2(VideoCamera())
    global flag
    global img
    

        


    if request.method == 'GET':
        print(request.args.get('get_value'))
        print(request.args.get('get_value'))

        if('get_value0' == request.args.get('get_value0')):
            res = request.args.get('get_value0')
            # image_capture()
            let = hoge.goodbye()
            print('get_value0')
            roc(VideoCamera())
            # return res
        elif(request.args.get('get_value0')):
            res = request.args.get('get_value1')
            # image_capture()
            let = hoge.goodbye()
            print('get_value1')
            # return res

    elif request.method == 'POST':
        print(request.form['post_value'])



        if("roc" == request.form['post_value']):
            res = request.form['post_value']
            # image_capture()
            # let = hoge.goodbye()
            # print('post_value')
            flag = 1
            # return res
        elif("noroc" == request.form['post_value']):
            res = request.form['post_value']
            # image_capture()
            # let = hoge.goodbye()
            # print('post_value')
            flag = 0
        elif("eval" == request.form['post_value']):
            res = request.form['post_value']
            # image_capture()
            # let = hoge.goodbye()
            # print('post_value')
            global result
            result = eval_script.main()
            spread(result)

        # res = request.form['post_value']

        # let = hoge.goodbye()
    
    while(flag==1):
        # os.rename("output.jpg", "output"+str(roc_frame_count+2)+".jpg")
        # frame = cv2.imread(img)
        # cropped_frame = img[20:360, 0:640]
        # cropped_frame = cv2.flip(cropped_frame,0)
        # # roc_frame_count+=1
        # out.write(cropped_frame)
        # out.release()

        frame_list.append(img)
        image = Image.open(io.BytesIO(frame_list[0])).convert("RGB")
        image.save('output3.jpg')
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    global result
    return resultf(test())
    # "/" を呼び出したときには、indexが表示される。


def gen(camera):
   
    while True:
        global img
        # global flag
        # if (flag):
        #     frame = camera.get_frame2()
        frame,img = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# returnではなくジェネレーターのyieldで逐次出力。
# Generatorとして働くためにgenとの関数名にしている
# Content-Type（送り返すファイルの種類として）multipart/x-mixed-replace を利用。
# HTTP応答によりサーバーが任意のタイミングで複数の文書を返し、紙芝居的にレンダリングを切り替えさせるもの。
# （※以下に解説参照あり）


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/eval_result')
def eval_result(result):
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
