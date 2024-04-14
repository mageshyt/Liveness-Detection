from ultralytics import  YOLO

############################
MODEL="models/yolov8n.pt"
OUTPUT_FOLDER="datasets/splitData"
BATCH_SIZE= 8
EPOCHS= 10
IMAGE_SIZE=416
############################
def main():
    # create the model
    model=YOLO(MODEL)
    # train the model
    model.train(data=f"{OUTPUT_FOLDER}/data.yaml", epochs=EPOCHS,device="mps",patience=5 )

    print("Training and evaluation completed!")
    # evaluate the model
    # modle.evaluate(data=f"{OUTPUT_FOLDER}/data.yaml")


if __name__ == '__main__':
    print("Training YOLO model...")
    main()




