
import numpy as np
import os
import systemcheck
import cv2
import torch

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

import torchvision.ops.boxes as bops
from statistics import mean
import time
from datetime import datetime

import face_recognition
from mailsend import sendmail

MODEL = "HOG"
TOLERANCE = 0.5
ID_TOLERANCE = 0.3
ID_FACE_TOLERANCE = 0.55

inp = input("Press Enter to run System or 'add' to Add new person:")
if 'add' in inp.lower():
    add_face = True
    print("New Person Enrolling Mode...\n\n")
    time.sleep(3)
else:
    add_face = False
    print("Normal Mode...\n\n")
    time.sleep(3)

def get_model(weights,  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    
    ###################### Load Model for Detection  #########################################  
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # print("Names:", names)

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    ###################### Model ready for Detection  #########################################  
    model_yolo = (device,model,stride, names, pt, jit, imgsz, half )
    print("Model ready for Detection")
    return model_yolo


def detect(model_yolo, 
        source, 
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        visualize = False,
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        conf_thres=0.20,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        ):

    device,model,stride, names, pt, jit, imgsz, half = model_yolo
     # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    for path, im, raw, vid_cap, s in dataset:
    
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
       
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

        coords = list()
        classes = list()
        # Process predictions
        
        
        for i, det in enumerate(pred):  # per image
            
            annotator = Annotator(raw, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to raw size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], raw.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # if names[c] in ['id card', 'person']: 
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    xyxy.append(conf)
                    coords.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]),float(xyxy[4])])
                    classes.append(names[c])
                  
            annotated_image = annotator.result()
        
    return annotated_image, coords, classes



def train_recogniser():
        
    print('Loading known faces for Retraining...')
    known_faces,known_names = [], []
    known_ids, known_id_names = [], []
    known_id_faces, known_id_faces_names = [], []

    KNOWN_FACES_DIR = 'data'

    #Training For FACE RECOGNITION
    for name in os.listdir(KNOWN_FACES_DIR):
        if "DS_Store" in name:
            continue
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}/face"):
            if "DS_Store" in filename:
                continue   
    
            print(f"{KNOWN_FACES_DIR}/{name}/face/{filename}", end = "")
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/face/{filename}")
           

            try:
                encoding = face_recognition.face_encodings(image, model = "hog")[0]
                known_faces.append(encoding)
                known_names.append(name)
                print(" --> Trained")
            except Exception as e:
                    print(e, 'Face not detected')
                    os.remove(f"{KNOWN_FACES_DIR}/{name}/face/{filename}")


    #Training For ID RECOGNITION
    for name in os.listdir(KNOWN_FACES_DIR):
        if "DS_Store" in name:
            continue
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}/id"):
            if "DS_Store" in filename:
                continue
    
            print(f"{KNOWN_FACES_DIR}/{name}/id/{filename}", end = "")
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/id/{filename}")
            try:
                id_encoding = face_recognition.face_encodings(image,[(0,image.shape[0], image.shape[1],0)])[0]
                known_ids.append(id_encoding)
                known_id_names.append(name)
                print(" --> Trained")
            except Exception as e:
                    print('ID not detected: ', e)
                    # os.remove(f"{KNOWN_FACES_DIR}/{name}/id/{filename}")


    #Training For FACE IN ID RECOGNITION
    for name in os.listdir(KNOWN_FACES_DIR):
        if "DS_Store" in name:
            continue
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}/id"):
            if "DS_Store" in filename:
                continue
    
            print(f"{KNOWN_FACES_DIR}/{name}/id/{filename}", end = "")
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/id/{filename}")

            try:
                encoding = face_recognition.face_encodings(image, model = "hog")[0]
                known_id_faces.append(encoding)
                known_id_faces_names.append(name)
                print(" --> Trained")
            except Exception as e:
                    print(e, ': Face not detected in ID')
                    

    # print("All Known Names:", set(known_names))
    np.save('known_faces', np.array(known_faces))
    np.save('known_names', np.array(known_names))

    np.save('known_ids', np.array(known_ids))
    np.save('known_id_names', np.array(known_id_names))

    np.save('known_id_faces', np.array(known_id_faces))
    np.save('known_id_faces_names', np.array(known_id_faces_names))



def get_main_face_coord(img, min_area = 1500):
    locations = face_recognition.face_locations(img, model=MODEL)
    main_face_coord = []
    main_face = []
    area = 0

    if len(locations) == 0:
        print("No Face Found")
        return main_face_coord, main_face, None

    for location in locations:

        # cv2.rectangle(image,(location[3],location[0]), (location[1], location[2]), (0,0,255), 2 )
        width = int(location[3] - location[1])
        height = int(location[0] - location[2])
        calc_area = width*height
        print("Face Area:", calc_area)
        if calc_area > area and calc_area > min_area:
            main_face_coord = list(location)
            area = calc_area
            print("Got Main Face", main_face_coord)

    try:
        if len(main_face_coord) > 0:
            if add_face:
                main_face_coord[3] -= 50 #start x
                main_face_coord[0] -= 100 #start Y

                main_face_coord[1] += 50 #end x
                main_face_coord[2] += 50 #end y

            
                if main_face_coord[3] < 0:
                    main_face_coord[3] = 0
                if main_face_coord[0] < 0:
                    main_face_coord[0] = 0


                if main_face_coord[2] > img.shape[0] :
                    main_face_coord[2] = img.shape[0] #height
    
                if main_face_coord[1] > img.shape[1] :
                    main_face_coord[1] = img.shape[1] #width


            print("Detected Main Face Coord:", main_face_coord)
            print(img.shape)

            main_face= img[main_face_coord[0]:main_face_coord[2], main_face_coord[3]:main_face_coord[1]]
            # cv2.imshow("Main Face", main_face)
    except Exception as e:
        print("Face Crop image Failed:", e)

    return main_face_coord, main_face, locations


if add_face:
    name = input("Enter Name:")
    roll = input("Enter Hall Ticket Number:")

    final_name = name+"["+roll+"]"
    try:
        os.mkdir(f"data")
    except:
        pass
    try:
        os.mkdir(f"data/{final_name}")
        os.mkdir(f"data/{final_name}/face")
        os.mkdir(f"data/{final_name}/id")
    except:
        print("Person Already Exists, Images will be Appended.")

else:

    print("Loading Available Faces...  ", end ="")
    train_recogniser()

    known_faces = np.load('known_faces.npy', allow_pickle=True).tolist()
    known_names = np.load('known_names.npy',allow_pickle=True).tolist()

    known_ids = np.load('known_ids.npy', allow_pickle=True).tolist()
    known_id_names = np.load('known_id_names.npy',allow_pickle=True).tolist()

    known_id_faces = np.load('known_id_faces.npy', allow_pickle=True).tolist()
    known_id_faces_names = np.load('known_id_faces_names.npy',allow_pickle=True).tolist()

    print("Done")
    print("Known Names:", set(known_names))


model_custom = get_model(weights= "best.pt")
model_person = get_model(weights= "yolov5m.pt")


cam = cv2.VideoCapture(1) #Make 0 as 1 for External Camera
for i in range(30):
    _, img = cam.read()
    cv2.waitKey(20)

count =0
noidcount = 0
while True:

    id_name, id_face_name, id_face_face_name, face_name = "","","",""
    _, img = cam.read()
    if _:
        count += 1
        if add_face and count > 200: #Give limited time to add face and ID
            break
        cv2.imwrite("live.png", img)
        image, custom_coords, custom_classes = detect(model_custom, source = f"live.png")
        if not add_face:
            image2, person_coords, person_classes = detect(model_person, source = f"live.png")

        final_class = []
        final_coords = []

        for cc in custom_classes:
            final_class.append(cc)
        if not add_face:
            for pc in person_classes:
                final_class.append(pc)

        for ctc in custom_coords:
            final_coords.append(ctc)
        if not add_face:
            for psc in person_coords:
                final_coords.append(psc)

        
        #DTECT ID CARD
        if 'id card' not in final_class:
            print("ID Card Not Found")
        for i in range(len(final_class)):
            if final_class[i] == 'id card':
                # print("GOT:", final_class[i])
                y_offset = 50
                id_image = image[final_coords[i][1]+y_offset:final_coords[i][3], final_coords[i][0]:final_coords[i][2]]
      
                if add_face: #Save ID
                    id_face_coord, id_face, id_face_locations = get_main_face_coord(id_image)
                    if id_face_locations is not None :
                        for face_location in id_face_locations:
                            if not add_face:
                                cv2.rectangle(image,(face_location[3]+final_coords[i][0],face_location[0]+final_coords[i][1]+y_offset), 
                            (face_location[1]+final_coords[i][0], face_location[2]+final_coords[i][1]+y_offset), 
                            (0,0,255), 3 )
                        
                        print("Saved ID")
                        cv2.putText(image, "ID Saved",(100,50),cv2.FONT_HERSHEY_PLAIN, 2, (255,100,100), 2)
                        cv2.imwrite(f"data/{final_name}/id/{str(time.time())[4:-7]}.png".replace("..","."),id_image)
                    else:
                        print("Make Sure Face is visible in ID")
                        cv2.putText(image, "FACE NOT VISIBLE IN ID",(100,50),cv2.FONT_HERSHEY_PLAIN, 2, (100,0,250), 2)

                else:
                    #Compare ID Cards
                    check_id_encodings = []
                    check_id_encodings.append(face_recognition.face_encodings(id_image,[(0,id_image.shape[0], id_image.shape[1],0)]))
                    check_id_encodings = np.array(check_id_encodings)

                    
                    for id_encoding in check_id_encodings:
                        id_results = face_recognition.compare_faces(known_ids, id_encoding, ID_TOLERANCE)

                    if True in id_results:
                        id_match = known_id_names[id_results.index(True)]
                        print("Found ID Match: ", id_match)
                        id_name = id_match


                    #Compare Face in ID Cards
                    id_face_coord, id_face, id_face_locations = get_main_face_coord(id_image)
                    id_face_encodings = face_recognition.face_encodings(id_image, id_face_locations)
                    if len(id_face_encodings) == 0:
                        print("No Face Visible in ID.")

                    if id_face_locations is not None and not add_face:
                        for face_location in id_face_locations:
                            cv2.rectangle(image,(face_location[3]+final_coords[i][0],face_location[0]+final_coords[i][1]+y_offset), 
                            (face_location[1]+final_coords[i][0], face_location[2]+final_coords[i][1]+y_offset), 
                            (0,0,255), 3 )

                        for id_face_encoding, id_face_location in zip(id_face_encodings, id_face_locations):
                            id_face_results = face_recognition.compare_faces(known_faces, id_face_encoding, ID_FACE_TOLERANCE)

                            if True in id_face_results:
                                id_face_match = known_names[id_face_results.index(True)]
                                print("Found ID Face Match1: ", id_face_match)
                                id_face_name = id_face_match


                    if id_face_locations is not None and not add_face:
                        for id_face_encoding, id_face_location in zip(id_face_encodings, id_face_locations):
                            id_face_results = face_recognition.compare_faces(known_id_faces, id_face_encoding, ID_FACE_TOLERANCE)

                            if True in id_face_results:
                                id_face_match = known_id_faces_names[id_face_results.index(True)]
                                print("Found ID Face Match2: ", id_face_match)
                                id_face_face_name = id_face_match
                    
                # cv2.imshow("ID Card", id_image)
            


        #Detect Main Face for Face Detection
        main_face_coord, main_face, all_face_locations = get_main_face_coord(img, min_area=20000)
        
        if len(main_face_coord)>0:
            if add_face:
                print("Saved Face")
                cv2.putText(image, "Face Saved",(100,100),cv2.FONT_HERSHEY_PLAIN, 2, (200,100,0), 2)
                cv2.imwrite(f"data/{final_name}/face/{str(time.time())[4:-7]}.png",main_face)

            #RECOGNISE ALL FACES
            encodings = face_recognition.face_encodings(img, all_face_locations)
            if all_face_locations is not None and not add_face:
                for face_encoding, face_location in zip(encodings, all_face_locations):
                    face_results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

                    if True in face_results:
                        face_match = known_names[face_results.index(True)]
                        print("Found Face Match: ", face_match)


            # RECOGNISE MAIN FACE
            main_face_encodings = []
            main_face_encodings.append(face_recognition.face_encodings(img,[(main_face_coord)]))
            main_face_encodings = np.array(main_face_encodings)
            if all_face_locations is not None and not add_face:
                for face_encoding, face_location in zip(main_face_encodings, [(main_face_coord)]):
                    face_results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                    if True in face_results:
                        face_match = known_names[face_results.index(True)]
                        print("Main Face Match: ", face_match)
                        face_name = face_match

            for face_location in all_face_locations:
                cv2.rectangle(image,(face_location[3],face_location[0]), (face_location[1], face_location[2]), (0,0,255), 3 )
            cv2.rectangle(image,(main_face_coord[3],main_face_coord[0]), (main_face_coord[1], main_face_coord[2]), (0,255,0), 3 )

        #PLOT ALL YOLO DETECTIONS
        for p in range(len(final_class)):
            if final_class[p] == "person" and not add_face:
                print("Plotting Person")
                color = (200, 0, 0) #BLUE
                start_point = (int(final_coords[p][0]), int(final_coords[p][1]))
                end_point = (int(final_coords[p][2]), int(final_coords[p][3]))
                image = cv2.rectangle(image, start_point, end_point, color, 2)
        
        for id in range(len(final_class)):
            if final_class[id] == "id card":
                print("Plotting ID Card")
                color = (150, 150, 255) #Pink
                start_point = (int(final_coords[id][0]), int(final_coords[id][1]))
                end_point = (int(final_coords[id][2]), int(final_coords[id][3]))
                image = cv2.rectangle(image, start_point, end_point, color, 2)
        
        for id in range(len(final_class)):
            if final_class[id] == "id card tag" and not add_face:
                print("Plotting ID Card Tags")
                color = (1, 100, 1) #Dark Green
                start_point = (int(final_coords[id][0]), int(final_coords[id][1]))
                end_point = (int(final_coords[id][2]), int(final_coords[id][3]))
                image = cv2.rectangle(image, start_point, end_point, color, 2)

        for id in range(len(final_class)):
            if final_class[id] == "tag" and not add_face:
                print("Plotting Tags")
                color = (1, 100, 100) #Dark Yello2
                start_point = (int(final_coords[id][0]), int(final_coords[id][1]))
                end_point = (int(final_coords[id][2]), int(final_coords[id][3]))
                image = cv2.rectangle(image, start_point, end_point, color, 2)


        if (len(face_name) > 0) and ((face_name == id_face_face_name) or (face_name == id_face_name) or (face_name == id_name)):
            print("Welcome:", face_name)
            cv2.putText(image, f"WELCOME {face_name}",(200,80),cv2.FONT_HERSHEY_PLAIN, 2, (100,255,100), 3)
            try:
                f1 = open("attendance.csv", 'x')
                f1.write("Date Time,Name RollNo\n")
            except:
                f1 = open("attendance.csv", 'a')
                
            
            f1.write(f"{str(datetime.now())[:-7]},{face_name}\n")
            f1.close()


            

        if 'person' in final_class and not add_face:
            if 'id card' not in final_class and 'id card tag' not in final_class:
                noidcount += 1
                if noidcount > 3:
                    print("Person without ID Card Found")
                    cv2.putText(image, f"Person without ID Card",(200,80),cv2.FONT_HERSHEY_PLAIN, 2, (0,5,250), 3)
                    
                    imgname = "voilation/"+"live"+str(int(time.time()))+".png"
                    cv2.imwrite(imgname, image)
                    sendmail(imgname)
            else:
                noidcount = 0
        else:
                noidcount = 0


        cv2.imshow("Output", image)
        cv2.waitKey(100)
        print("#"*20)
