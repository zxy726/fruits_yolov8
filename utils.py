from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image,ImageDraw,ImageFont
import tempfile
import config

def _display_detected_frames(model, st_frame, image):
    """
    使用YOLOv8模型在视频帧上显示检测到的对象。
    :param model (YOLOv8): 一个包含YOLOv8模型的`YOLOv8`类的实例。
    :param st_frame (Streamlit对象): 一个Streamlit对象，用于显示检测到的视频。
    :param image (numpy数组): 一个表示视频帧的numpy数组。
    :return: None
    """
    # 将图像调整为标准大小
    # image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用YOLOv8模型预测图像中的对象
    # res = model.predict(image)

    # 在视频帧上绘制检测到的对象

    img_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    res_plotted = predict_image(model,img_rgb)
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def predict_image(model,uploaded_image):
    res = model.predict(uploaded_image)
    labels = res[0].names
    probs = res[0].probs.cpu().numpy()
    top5_index_list = probs.top5
    top5_confidence = probs.top5conf
    uploaded_image = uploaded_image.convert('RGB')
    draw = ImageDraw.Draw(uploaded_image)
    try:
        font = ImageFont.truetype(str(config.FONT_DIR), 32)
    except:
        raise "font目录或者font目录下SimHei.ttf 字体文件不存在!"
    for i in range(len(top5_index_list)):
        class_name = labels[top5_index_list[i]]
        confidence = top5_confidence[i] * 100
        text = '{:<5} {:>.2f}%'.format(class_name, confidence)
        draw.text((5, 30 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
    return uploaded_image
def infer_uploaded_image(model):
    """
    执行上传图像的推理
    :param model: 一个包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="选择一张图片...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            #将上传的图片添加到页面并加上标题。
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("执行中..."):
                uploaded_image = predict_image(model, uploaded_image)
                with col2:
                    st.image(uploaded_image,
                             caption="Detected Image",
                             use_column_width=True)

def infer_uploaded_video(model):
    """
    执行上传视频的推理
    :param model: 一个包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """

    source_video = st.sidebar.file_uploader(
        label="选择一个视频..."
    )
    col1, col2 = st.columns(2)
    with col1:
        if source_video:
            st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("执行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    with col2:
                        while (vid_cap.isOpened()):
                            success, image = vid_cap.read()
                            if success:
                                _display_detected_frames(
                                    model,
                                    st_frame,
                                    image
                                )
                            else:
                                vid_cap.release()
                                break
                except Exception as e:
                     print(e)
                     st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(model):
    """
    执行摄像头推理
    :param model: 一个包含YOLOv8模型的`YOLOv8`类的实例。
    :return: None
    """

    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
