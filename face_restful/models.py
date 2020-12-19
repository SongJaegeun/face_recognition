from django.db import models
# from django.contrib.auth.models import User
# Create your models here.


class User(models.Model):

    user_id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=30)
    gender = models.CharField(max_length=15)
    age = models.IntegerField()
    phone = models.CharField(max_length=15)
    address = models.CharField(max_length=50)

    class Meta:
        db_table = 'USER_TB'

    def __str__(self):
        return self.user_name


class Order(models.Model):
    order_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField()
    menu_name = models.CharField(max_length=30)
    category = models.CharField(max_length=15)
    order_date = models.DateField()
    point = models.IntegerField()
    order_no = models.IntegerField()

    class Meta:
        db_table = 'ORDER_TB'

    def __str__(self):
        return self.menu_name
