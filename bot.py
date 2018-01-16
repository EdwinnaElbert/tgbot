# -*- coding: utf-8 -*-
import config
import telebot as tb
import requests
import numpy as np
import cv2

bot = tb.TeleBot(config.token)
# TODO play with tg bot
# TODO try paraller handling images
@bot.message_handler(content_types=["photo"])
def repeat_all_messages(message):
  photo_id = message.photo[3].file_id # index specifies size of an image
  # download_file and save it
  file_info = bot.get_file(photo_id)
  downloaded_file = bot.download_file(file_info.file_path)
  with open('saved_images/new_file','wb') as new_file:
    new_file.write(downloaded_file)

# opencv work
# face_cascade = cv2.CascadeClassifier('/home/edwinna/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')

# TODO get pipe sign and catch extension
if new_file != null :
  img = cv2.imread('saved_images/new_file.png') or cv2.imread('saved_images/new_file.jpg')
  img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)

# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# cropped = []
# for (x,y,w,h) in faces:
#   cropped = img[y:y+h, x:x+w]
#
# cv2.imwrite('detected_faces/new_file.png', cropped)

# opencv video work

  cap = cv2.VideoCapture('videos/phone.mp4')
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('saved_videos/video.avi',fourcc, 20.0, (640,480))
# cap = cv2.VideoCapture(0)

  while(True):
  # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
  # Our operations on the frame come here
  # gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

  # hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

  # lower_range = np.array([79, 115, 70])
  # upper_range = np.array([122, 148, 92])

  #lower_range = np.array([60, 140, 60])
  # upper_range = np.array([122, 230, 100])

  # lower_range = np.array([40, 140, 40])
  # upper_range = np.array([130, 255, 100])

  # lower_range = np.array([33, 130, 28])
  # upper_range = np.array([130, 255, 100])

  # lower_range = np.array([0, 120, 0])
  # upper_range = np.array([130, 255, 100])

  # TODO after getting video catch green
    lower_range = np.array([0, 120, 0])
    upper_range = np.array([140, 255, 115])

    mask = cv2.inRange(frame, lower_range, upper_range)

  # Display the resulting frame
    ret, thresh_img = cv2.threshold(mask, 91, 255, cv2.THRESH_BINARY)

    contours =  cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
  # for c in contours:
#     cv2.drawContours(frame, [c], -1, (255,0,0), 3)

    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])
  # img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
    pts1 = np.float32([[640,0], [0,480], [0,0], [640,480]])
    pts2 = np.float32([[extLeft[0], extLeft[1]], [extRight[0], extRight[1]], [extTop[0], extTop[1]], [extBottom[0], extBottom[1]]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img2 = cv2.warpPerspective(img,M,(640,480))

    mask_inv = cv2.bitwise_not(mask)
  # Now black-out the area mask in video
    video_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)
  # Take only region of image.
    img_fg = cv2.bitwise_and(img2, img2, mask = mask)

    dst = cv2.add(video_bg, img_fg)
  # cv2.circle(dst, tuple(extLeft), 3, (0,0,255))
  # cv2.circle(dst, tuple(extRight), 3, (0,0,255))
  # cv2.circle(dst, tuple(extTop), 3, (0,0,255))
  # cv2.circle(dst, tuple(extBottom), 3, (0,0,255))

  # cv2.imshow('frame', dst)
    out.write(dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
  cap.release()
  out.release()
  cv2.destroyAllWindows()

# bot.send_photo(chat_id=message.chat.id, photo=open('detected_faces/new_file.png', 'rb'))
bot.send_video(chat_id=message.chat.id, video=open('saved_videos/video.avi', 'rb'))

if __name__ == '__main__':
  bot.polling(none_stop=True)
