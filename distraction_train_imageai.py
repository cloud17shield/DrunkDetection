from imageai.Prediction.Custom import ModelTraining
model_trainer = ModelTraining()
model_trainer.setModelTypeAsSqueezeNet()
model_trainer.setDataDirectory("/Users/ranxin/Downloads/state-farm-distracted-driver-detection/imgs")
model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
