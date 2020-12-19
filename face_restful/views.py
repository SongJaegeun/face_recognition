from django.shortcuts import render
from rest_framework import viewsets
from .serializers import UserSerializer
from .models import User,Order
from rest_framework.response import Response
from rest_framework.decorators import api_view
import numpy as np
import pandas as pd
import cv2
import dlib
from django.db import connections
from scipy.sparse.linalg import svds
from django.http import JsonResponse
import requests, json

descs = np.load('./static/descs.npy', allow_pickle=True)
descs = np.array(descs).reshape(1,)[0]
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./static/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./static/deep_face_inception_v1.dat')
queryset = Order.objects.all()
query, params = queryset.query.as_sql(compiler='sql_server.pyodbc.compiler.SQLCompiler', connection=connections['default'])
order_df = pd.read_sql_query(query, connections['default'], params=params)


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


def recommend(user_id):


    group_data = order_df.groupby(['category'])

    main_menu_group = group_data.get_group('메인')
    side_menu_group = group_data.get_group('사이드')

    user_menu_rating = main_menu_group.pivot_table(
        'point',
        index='user_id',
        columns='menu_name').fillna(0)



    # matrix는 위에서만든 pivot_table값을 numpy matrix로 만듬
    matrix = user_menu_rating.values

    # user_rating_avg는 사용자의 평균 맛평가
    user_rating_avg = np.mean(matrix, axis=1)

    # matrix_user_avg : 사용자-평가에대해 사용자 평균 평점을 뺀것
    matrix_user_avg = matrix - user_rating_avg.reshape(-1, 1)


    # scipy에서제공하는 svd
    # U행렬, sigma행렬, V전치행렬 반환
    U, sigma, Vt = svds(matrix_user_avg, k=2)

    sigma = np.diag(sigma)

    # matrix_user_avg를 svd를 적용해 분해한 상태이므로 원복 필요
    svd_user_predicted_rating = np.dot(np.dot(U, sigma), Vt) + user_rating_avg.reshape(-1, 1)

    svd_preds = pd.DataFrame(svd_user_predicted_rating,
                             index=user_menu_rating.index,
                             columns=user_menu_rating.columns)

    def recommend_menu(svd_preds, og_user_id, og_order_item, og_point, num_recommendations=5):
        # 맛평가 평점이 높은 순으로 정렬
        sort_user_predictions = svd_preds.index.sort_values(ascending=False)

        # 원본데이터에서 user_id에 해당하는 데이터 뽑기
        user_data = og_point[og_point.user_id == og_user_id]

        # 원본데이터에서 사용자가 주문했던 음식은 제외한 데이터 추출
        recommendations = og_order_item[-og_order_item['menu_name'].isin(user_data['menu_name'])]
        # 사용자의 메뉴 평점이 높은 순으로 정렬된 데이터와 recommendations를 합친다
        recommendations = recommendations.merge(pd.DataFrame(sort_user_predictions).reset_index(), on='user_id')
        # 컬럼이름을 바꾸고 정렬해서 return
        recommendations = recommendations.rename(columns={'index': 'Predictions'}).sort_values('Predictions',
                                                                                               ascending=False)

        return user_data, recommendations

    already_rated_main, predictions_main = recommend_menu(svd_preds, user_id, main_menu_group,
                                                          main_menu_group, 10)
    already_rated_main = already_rated_main.sort_values(by='order_no', ascending=False)

    already_rated_side, predictions_side = recommend_menu(svd_preds, user_id, side_menu_group,
                                                          side_menu_group, 10)
    already_rated_side = already_rated_side.sort_values(by='order_no', ascending=False)

    main_recommend = predictions_main['menu_name'].drop_duplicates().values
    side_recommend = predictions_side['menu_name'].drop_duplicates().values

    return main_recommend, side_recommend

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

        return Response(f'{request.data["user_name"]}님 등록 완료')

@api_view(('GET', 'POST'))
def signpicture(request):
    if request.method == 'GET':
        query_set = User.objects.all()
        serializer = UserSerialiㅎ(query_set, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':

        user_id = User.objects.last().user_id
        new_photo = request.FILES.get('photo')
        image= new_photo.read()
        with open('./static/save3.jpg', 'wb') as f:
            f.write(image)
        photo_arr = load_img({user_id: './static/save3.jpg'})
        descs[user_id] = photo_arr[user_id]

        np.save('./static/descs.npy', descs)

        return Response(f'{request.data["user_name"]}님 등록 완료')


@api_view(('GET', 'POST'))
def face_recognition(request):
    if request.method == 'GET':
        query_set = User.objects.all()
        serializer = UserSerializer(query_set, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        photo = request.FILES.get('photo')
        image = photo.read()
        with open('./static/save.jpg', 'wb') as f:
            f.write(image)

        img_bgr = cv2.imread('./static/save.jpg')
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

        recommends = recommend(user_id)
        URL = ''  # url 입력
        params = {{'user_name': user_id, 'main_recommend': recommends[0], 'side_recommend': recommends[1]}}
        requests.get(URL, params=params)

        return Response({'user_name': user_id, 'main_recommend': recommends[0], 'side_recommend': recommends[1]})


