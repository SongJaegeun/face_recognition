from django.shortcuts import render
from rest_framework import viewsets
from .serializers import UserSerializer
from .models import User
from rest_framework.response import Response
from rest_framework.decorators import api_view
import numpy as np
import cv2
import dlib
from django.db import connection



descs = np.load('./static/descs.npy', allow_pickle=True)
descs = np.array(descs).reshape(1,)[0]
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./static/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./static/deep_face_inception_v1.dat')


def find_faces(img):  # 얼굴 추출 및 랜드마크 검출
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects = []
    shapes = []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)

    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np


def encode_faces(img, shapes):  # 128차원의 벡터 추출
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


def load_img(img_paths):
    descs = {key: None for key in img_paths.keys()}
    for name, img_path in img_paths.items():
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        _, img_shapes, _ = find_faces(img_rgb)
        descs[name] = encode_faces(img_rgb, img_shapes)[0]

    return descs



@api_view(('GET', 'POST'))
def signin(request):
    if request.method == 'GET':
        query_set = User.objects.all()
        serializer = UserSerializer(query_set, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        create_user = User(user_name=request.data['user_name'],
                           gender=request.data['gender'], age=request.data['age'], phone=request.data['phone'],
                           address=request.data['address'])
        create_user.save()
        user_id = User.objects.last().user_id
        # new_photo = request.POST['photo']
        # photo_arr = load_img({user_id: './test/p.jpg'})
        # descs[user_id] = photo_arr[user_id]
        # np.save('./static/descs.npy', descs)

        return Response(f'{request.data["user_name"]}님 등록 완료')


@api_view(('GET', 'POST'))
def face_recognition(request):

    if request.method == 'GET':
        query_set = User.objects.all()
        serializer = UserSerializer(query_set, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        # photo = request.POST.get()
        photo = './static/1.jpg'
        img_bgr = cv2.imread(photo)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rects, shapes, _ = find_faces(img_rgb)
        descroptors = encode_faces(img_rgb, shapes)

        for i, desc in enumerate(descroptors):
            min_dist = 100
            min_id = None
            for name, saved_desc in descs.items():
                dist = np.linalg.norm([desc] - saved_desc, axis=1)  # 유클리디안 거리

                if dist <= min_dist:
                    min_dist = dist
                    min_id = name

            user_id = min_id

        query_set = User.objects.filter(user_id=user_id)
        serializer = UserSerializer(query_set, many=True)


        return Response(serializer.data)


