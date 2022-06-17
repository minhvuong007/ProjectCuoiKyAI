from tkinter import*
import cv2
from PIL import Image, ImageTk
import time

import numpy as np
import tensorflow as tf
from keras.models import load_model

#=================================

physical_devices = tf.config.list_physical_devices("CPU")
threshold = 0.75

# load file mô hình mạng nơ-ron
model = load_model('model.h5')

#=============================
class TestVideo:
    def __init__(self, video_source = 0):
        self.appName = "MyVideo"  
        self.window = Tk()  
        self.window.title(self.appName)
        self.window.geometry("600x400")
        #self.window.resizable(0,0)
        self.window.configure(bg = 'white')

        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        #Top frame
        self.Top_frame = Frame(self.window, bg = "white", width = 600, height = 70)
        self.Top_frame.place(x = 0, y= 0)

        #logo
        image_1 = Image.open("F:\\TKINTER\\project_AI\\traffic_sign.png")
        resize_image = image_1.resize((70,70), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(resize_image)
       
        self.label_1 = Label(self.Top_frame, image = img)
        self.label_1.place(x = 0, y = 0)

        self.label_2 = Label(self.Top_frame, text = "TRAFFIC SIGN RECOGNITION", font = "arial 20 bold", bg = 'white', fg = 'black')
        self.label_2.place(x = 100, y = 22.5)

       # Button
        self.btn_capture = Button(self.window, text = "Recognition", font = 15, command = self.predict)
        self.btn_capture.place(x = 400 , y = 300)

        # Text
        self.text_1 = Text(self.window, bg = "white", font ="arial 13 bold")
        self.text_1.place(x = 370, y= 120, width = 200, height = 30)

        self.text_2 = Text(self.window, bg = "white", font ="arial 13 bold")
        self.text_2.place(x = 370, y= 200, width = 200, height = 30)

        # Create a canvas
        self.canvas_1 = Canvas(self.window, width = self.vid.width,height = self.vid.height, bg= 'white')
        self.canvas_1.place(x = 0, y = 115)

        self.update()
        self.window.mainloop()
        self.predict()

    def update(self):
        # Lấy khung ảnh
        isTrue, frame = self.vid.getFrame()
        #frame = cv2.flip(frame, 1)
        if isTrue:             
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas_1.create_image(0, 0, image = self.photo, anchor = NW)
        self.window.after(15, self.update)

    def predict(self):
        isTrue, frame = self.vid.getFrame()
        #frame = cv2.flip(frame, 1)
        if isTrue:             
            img = np.asarray(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            img = cv2.resize(img,(32,32))
            img = img/255
            img = img.reshape(1,32,32,1)
        
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)                 
            
            if probabilityValue > threshold:
                print(self.getClassName(classIndex))
                self.text_1.delete('1.0', END)
                self.text_1.insert('1.0', self.getClassName(classIndex)+"\n")

                self.text_2.delete('1.0', END)
                self.text_2.insert('1.0',"Độ chính xác: " + str(round(probabilityValue*100, 2)) + " %"+"\n")              
            
        
    def getClassName(self,classNo):
        if classNo == 0:
            return 'Gioi han 20km/h'
        elif classNo == 1:
            return 'Gioi han 30km/h'
        elif classNo == 2:
            return 'Gioi han 50km/h'
        elif classNo == 3:
            return 'Gioi han 60km/h'
        elif classNo == 4:
            return 'Gioi han 70km/h'
        elif classNo == 5:
            return 'Gioi han 80km/h'
        elif classNo == 6:
            return 'Ket thuc gioi han 80km/h'
        elif classNo == 7:
            return 'Gioi han 100km/h'
        elif classNo == 8:
            return 'Gioi han 120km/h'
        elif classNo == 9:
            return 'Cam vuot'
        elif classNo == 10:
            return 'Cam xe vuot qua 3.5 tan'
        elif classNo == 11:
            return 'Right-of-way at the next intersection'
        elif classNo == 12:
            return 'Duong uu tien'
        elif classNo == 13:
            return 'Yield'
        elif classNo == 14:
            return 'STOP'
        elif classNo == 15:
            return 'Khong co phuong tien luu thong'
        elif classNo == 16:
            return 'Cam xe vuot qua 3.5 tan'
        elif classNo == 17:
            return 'Cam vao'
        elif classNo == 18:
            return 'General caution'
        elif classNo == 19:
            return 'Duong quanh co nguy hiem ben trai'
        elif classNo == 20:
            return 'Duong quanh co nguy hiem ben phai'
        elif classNo == 21:
            return 'Double curve'
        elif classNo == 22:
            return 'Duong gap genh'
        elif classNo == 23:
            return 'Duong tron truot'
        elif classNo == 24:
            return 'Duong bi thu hep ben phai'
        elif classNo == 25:
            return 'Cong truong'
        elif classNo == 26:
            return 'Tin hieu giao thong'
        elif classNo == 27:
            return 'Cam nguoi di bo'
        elif classNo == 28:
            return 'Tre em qua duong'
        elif classNo == 29:
            return 'Xe dap qua duong'
        elif classNo == 30:
            return 'Can than bang tuyet'
        elif classNo == 31:
            return 'Dong vat qua duong'
        elif classNo == 32:
            return 'End of all speed and passing limits'
        elif classNo == 33:
            return 'Re phai o phia truoc'
        elif classNo == 34:
            return 'Re trai o phia truoc'
        elif classNo == 35:
            return 'Di thang'
        elif classNo == 36:
            return 'Di thang hoac re phai'
        elif classNo == 37:
            return 'Di thang hoac re trai'
        elif classNo == 38:
            return 'Di ben phai'
        elif classNo == 39:
            return 'Di ben trai'
        elif classNo == 40:
            return 'Di vong'
        elif classNo == 41:
            return 'End of no passing'
        elif classNo == 42:
            return 'End of no passing by vechiles over 3.5 metric tons'       

class MyVideoCapture:
    def __init__(self, video_source):
        # Mở video
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Không thể mở Camera\n", video_source)

        # Lấy kích thước khung video
        # Set kích thước khung video
        self.vid.set(3, 250)
        self.vid.set(4, 250)
        self.width = self.vid.get(3)
        self.height = self.vid.get(4)

    def getFrame(self):
        if self.vid.isOpened():
            isTrue, frame = self.vid.read()
            if isTrue:
                    # Video hiện ra có màu
                return (isTrue, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (isTrue, None)            
        else:
            return (isTrue, None)
        
    def __def__(self):
        if self.vid.isOpened():
            self.vid.release()
        
if __name__ == "__main__":
    TestVideo()


