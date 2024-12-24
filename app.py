import streamlit as st
import math
import random
import cvzone
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
import speech_recognition as sr
import time
import base64

# 设置页面布局
st.set_page_config(
    page_title="YOLOv8 & Snake Game",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置透明背景的 CSS 样式
def set_sidebar_transparent():
    st.markdown(
        """
        <style>
        /* 设置侧边栏背景为透明 */
        .css-1d391kg {
            background-color: rgba(0, 0, 0, 0) !important;
        }

        /* 可选：设置侧边栏文本颜色 */
        .css-1d391kg .stSidebar > div > div {
            color: #000000;  /* 文本颜色 */
        }

        /* 可选：设置按钮背景为透明 */
        .css-1d391kg .stButton > button {
            background-color: rgba(0, 0, 0, 0) !important;
            color: #F39C12;  /* 按钮文字颜色 */
            border: 1px solid #F39C12;  /* 按钮边框颜色 */
        }
        </style>
        """, unsafe_allow_html=True
    )

# 在 Streamlit 页面中应用自定义样式
set_sidebar_transparent()

# 添加背景图片的 CSS 样式
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 调用函数添加背景图片
add_bg_from_local('background/3.jpg')  # 请确保背景图片放在与你的脚本相同的目录下

# 主页面标题
st.title("基于YOLOv8的水果分类系统 & 贪吃蛇游戏")

# 会话状态：跟踪登录状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


# 语音识别功能
def voice_recognition():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        st.info("请说：'芝麻开门' 以进入识别系统")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        recognized_text = recognizer.recognize_google(audio, language="zh-CN")
        st.write(f"您说了: {recognized_text}")
        if "芝麻开门" in recognized_text:
            st.session_state.logged_in = True
            st.sidebar.success("语音识别成功！欢迎进入识别系统。")
        else:
            st.sidebar.error("识别失败，请再试一次。")
    except sr.UnknownValueError:
        st.sidebar.error("无法理解您的语音，请再试一次。")
    except sr.RequestError as e:
        st.sidebar.error(f"无法连接到语音识别服务; {e}")


# 语音识别页面
if not st.session_state.logged_in:
    st.sidebar.header("请使用语音识别登录")
    if st.sidebar.button("开始识别"):
        voice_recognition()
else:
    st.sidebar.success("您已登录。")
    # 登录成功后显示“开始游戏”按钮
    if st.sidebar.button("开始游戏"):
        # 开始游戏后执行贪吃蛇游戏逻辑
        st.sidebar.write("游戏已启动！")

        cap = cv2.VideoCapture(0)  # 使用摄像头
        cap.set(3, 1280)
        cap.set(4, 720)

        detector = HandDetector(detectionCon=0.8, maxHands=1)


        class SnakeGameClass:
            def __init__(self, pathFood):  # 构造方法
                self.points = []  # 蛇的身体点
                self.lengths = []  # 每两个点之间的长度
                self.currentLength = 0  # 蛇的总长度
                self.allowedLength = 150  # 蛇允许的最大长度
                self.previousHead = 0, 0  # 第二个头结点

                self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
                self.hFood, self.wFood, _ = self.imgFood.shape
                self.foodPoint = 0, 0
                self.randomFoodLocation()

                self.score = 0
                self.gameOver = False

            def randomFoodLocation(self):
                self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

            def update(self, imgMain, currentHead):  # 实例方法
                if self.gameOver:
                    cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                                       scale=7, thickness=5, offset=20)
                    cvzone.putTextRect(imgMain, f'Your Score:{self.score}', [300, 550],
                                       scale=7, thickness=5, offset=20)
                else:
                    px, py = self.previousHead
                    cx, cy = currentHead

                    self.points.append([cx, cy])  # 将食指尖坐标添加到蛇身点列表
                    distance = math.hypot(cx - px, cy - py)  # 计算两点之间的距离
                    self.lengths.append(distance)  # 添加到蛇的长度列表
                    self.currentLength += distance
                    self.previousHead = cx, cy

                    # 蛇的长度控制
                    if self.currentLength > self.allowedLength:
                        for i, length in enumerate(self.lengths):
                            self.currentLength -= length
                            self.lengths.pop(i)
                            self.points.pop(i)
                            if self.currentLength < self.allowedLength:
                                break

                    # 检查蛇是否吃到食物
                    rx, ry = self.foodPoint
                    if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                            ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                        self.randomFoodLocation()
                        self.allowedLength += 50
                        self.score += 1
                        print(self.score)

                    # 绘制蛇
                    if self.points:
                        for i, point in enumerate(self.points):
                            if i != 0:
                                cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                        cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

                    # 绘制食物
                    imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                                (rx - self.wFood // 2, ry - self.hFood // 2))

                    cvzone.putTextRect(imgMain, f'Your Score:{self.score}', [50, 80],
                                       scale=3, thickness=5, offset=10)

                    # 检查是否碰到墙壁
                    pts = np.array(self.points[:-2], np.int32)
                    pts = pts.reshape((-1, 1, 2))  # 重塑为一个形状为(-1, 1, 2)的矩阵
                    cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
                    minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

                    if -1 <= minDist <= 1:
                        print("撞墙了")
                        self.gameOver = True
                        self.points = []  # 重置蛇的身体点
                        self.lengths = []
                        self.currentLength = 0
                        self.allowedLength = 150
                        self.previousHead = 0, 0
                        self.randomFoodLocation()

                return imgMain


        game = SnakeGameClass("apple.png")

        # 处理每一帧图像
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)  # 翻转图像，使其成为镜像
            hands, img = detector.findHands(img, flipType=False)

            if hands:
                lmList = hands[0]['lmList']  # 获取手势关键点列表
                pointIndex = lmList[8][0:2]  # 获取食指尖的x、y坐标
                img = game.update(img, pointIndex)

            cv2.imshow("Snake Game", img)
            key = cv2.waitKey(1)
            if key == ord('r'):
                game.gameOver = False

        cap.release()
        cv2.destroyAllWindows()

    else:
        # 水果分类任务
        st.sidebar.header("模型配置")

        model_type = st.sidebar.selectbox(
            "选取模型",
            config.Classification_MODEL_LIST
        )

        model_path = ""
        if model_type:
            model_path = Path(config.Classification_MODEL_DIR, str(model_type))
        else:
            st.error("请选择模型")

        # 加载预训练深度学习模型
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"无法加载模型，请检查路径：{model_path}")

        # 图片/视频配置
        st.sidebar.header("图片/视频配置")
        source_selectbox = st.sidebar.selectbox(
            "选取文件类型",
            config.SOURCES_LIST
        )

        if source_selectbox == config.SOURCES_LIST[0]:  # 图片
            infer_uploaded_image(model)
        elif source_selectbox == config.SOURCES_LIST[1]:  # 视频
            infer_uploaded_video(model)
        elif source_selectbox == config.SOURCES_LIST[2]:  # 摄像头
            infer_uploaded_webcam(model)
        else:
            st.error("当前只支持'图片'和'视频'来源")