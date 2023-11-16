print(1)
import pickle
print(2)
from keras.models import load_model
print(3)
from keras.preprocessing.sequence import pad_sequences
print(4)
import numpy as np
print(5)
import sys
print("imported all modules")

def load_tokenizer_and_encoder(tokenizer_path, encoder_path):
	with open(tokenizer_path, 'rb') as file:
		tokenizer = pickle.load(file)
	with open(encoder_path, 'rb') as file:
		encoder = pickle.load(file)
	return tokenizer, encoder

def predict_class(text, model_path, tokenizer, encoder):
	text = tokenizer.texts_to_sequences([text])
	text = pad_sequences(text)

	model = load_model(model_path)

	prediction = model.predict(text,verbose=False)

	predicted_class = np.argmax(prediction, axis=-1)
	confidence = np.max(prediction, axis=-1)

	predicted_class = np.take(encoder.classes_, predicted_class)

	return predicted_class[0], confidence[0]

def predict_classes(text, model_path, tokenizer, encoder):
	text = tokenizer.texts_to_sequences([text])
	text = pad_sequences(text)

	model = load_model(model_path)

	prediction = model.predict(text,verbose=False)

	top3_predicted_classes_indices = np.argpartition(prediction[0], -3)[-3:]
	top3_confidences = prediction[0][top3_predicted_classes_indices]
	top3_predicted_classes = encoder.classes_[top3_predicted_classes_indices]

	return list(zip(top3_predicted_classes, top3_confidences))

sub_tokenizer, sub_encoder = load_tokenizer_and_encoder('./Grader/sub-tokenizer.pickle', './Grader/sub-encoder.pickle')
tokenizer, encoder = load_tokenizer_and_encoder('tokenizer.pickle', 'encoder.pickle')

print("finished building functions")

text = "Tôi đang vật lộn với tất cả. Xin lỗi vì tiếng Anh của tôi, tôi là người Ý. Tôi ghét công việc của mình. Tôi ghét những cơn nghiện của mình. Tôi đấu tranh với cái nhìn của mình. 22, 1,72 m, tóc tôi mỏng đi và tôi sắp bị hói. Tôi có 2 bạn gái nhưng biết tôi không thể nói chuyện với một cô gái. Tôi luôn có rất nhiều quầng thâm dưới mắt vì tôi hút quá nhiều cỏ dại khiến tôi cảm thấy mình thật ngu ngốc. Tôi lo lắng về việc nói chuyện với bất cứ ai chưa phải là bạn của tôi. Mỗi ngày đều giống nhau. Tôi chỉ hạnh phúc vào cuối tuần khi tôi có thể theo đuổi sở thích của mình. Tôi biết một số người có thể có một tình huống tồi tệ hơn, nhưng tôi phải chịu đựng rất nhiều mỗi ngày. Tôi đi ngủ muộn vào ban đêm vì tôi không muốn đối mặt với ngày hôm sau. Tôi đã trị liệu được 2 năm, nhà tâm lý học giúp tôi nhưng tôi không thể thay đổi. Tôi mệt"

if text == "":
	print("No input provided")
	sys.exit()
classes_and_confidences = predict_classes(text, 'MID-l.h5', tokenizer, encoder)
print(f"text: {text}")
for predicted_class, confidence in sorted(classes_and_confidences, key=lambda x: x[1], reverse=True):
	print(f'Predicted problems: {predicted_class}, Confidence: {confidence}')
sub_class,sub_confidence = predict_class(text, './Grader/sub-MID-l.h5', sub_tokenizer, sub_encoder)
print(f'Predicted severity: {sub_class}, Confidence: {sub_confidence}')