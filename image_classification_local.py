from imageai.Prediction import ImagePrediction
import cv2
import imutils

prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel(prediction_speed='fastest')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    predictions, probabilities = prediction.predictImage(image_input=frame, result_count=3, input_type="array")
    for i in range(len(predictions)):
        cv2.putText(frame, str(predictions[i]) + ':' + str(probabilities[i]), (10, 30 + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
