import tkinter as tk
from tkinter import messagebox
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import threading
from time import sleep


root = tk.Tk()

label_status = tk.Label(root, text='Status: Started', bg='#2d3145', fg='#00ff00')


model = WhisperModel("medium")

class Recorder:
	def __init__(self):
		self.is_recording = False
		self.recording = []
		self.thread = None

	def record(self):
		texts = ""
		show_box = ""
		if not self.is_recording:
			self.is_recording = True
			self.thread = threading.Thread(target=self._record)
			self.thread.start()
		else:
			self.is_recording = False
			self.thread.join()
			sf.write('my_Audio.mp3', self.recording, 44100)
			segments, info = model.transcribe("my_Audio.mp3")
			for segment in segments:
				print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
				texts = f"{texts} {segment.text}"
			c_and_c = predict_classes(texts, 'MID-l.h5', tokenizer, encoder)
			sub_class,sub_confidence = predict_class(texts, './Grader/sub-MID-l.h5', sub_tokenizer, sub_encoder)
			for predicted_class, confidence in sorted(c_and_c, key=lambda x: x[1], reverse=True):
				show_box += (f'Vấn đề được dự đoán: {str(predicted_class)}, Khả năng chuẩn đoán: {round(confidence*100, 3)}%'+"\n")
			messagebox.showinfo("Chuẩn đoán", show_box)
			if sub_class == 1:
				sub_class = "Nhẹ"
			elif sub_class == 2:
				sub_class = "Vừa"
			else:
				sub_class = "Nặng"
			messagebox.showinfo("Mức nghiêm trọng",f'Mức nghiêm trọng được dự đoán: {sub_class}, Độ chính xác: {round(sub_confidence*100, 3)}%')

	def _record(self):
		self.recording = sd.rec(int(44100 * 30), samplerate=44100, channels=1)
		while self.is_recording:
			sd.sleep(1000)
		sd.stop()

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
	print("laoded model(s)")

	prediction = model.predict(text)

	predicted_class = np.argmax(prediction, axis=-1)
	confidence = np.max(prediction, axis=-1)

	predicted_class = np.take(encoder.classes_, predicted_class)

	return predicted_class[0], confidence[0]

def predict_classes(text, model_path, tokenizer, encoder):
	text = tokenizer.texts_to_sequences([text])
	text = pad_sequences(text)

	model = load_model(model_path)
	print("loaded model(s)")

	prediction = model.predict(text,verbose=False)

	top3_predicted_classes_indices = np.argpartition(prediction[0], -3)[-3:]
	top3_confidences = prediction[0][top3_predicted_classes_indices]
	top3_predicted_classes = encoder.classes_[top3_predicted_classes_indices]

	return list(zip(top3_predicted_classes, top3_confidences))

sub_tokenizer, sub_encoder = load_tokenizer_and_encoder('./Grader/sub-tokenizer.pickle', './Grader/sub-encoder.pickle')
tokenizer, encoder = load_tokenizer_and_encoder('tokenizer.pickle', 'encoder.pickle')
print("loaded tkenizers and encoders")
recorder = Recorder()

def predict():
	global label_status
	text = entry.get()
	if text != "":
		show_box = ""
		c_and_c = predict_classes(text, 'MID-l.h5', tokenizer, encoder)
		sub_class,sub_confidence = predict_class(text, './Grader/sub-MID-l.h5', sub_tokenizer, sub_encoder)
		label_status.configure(text='Status: processed.')
		for predicted_class, confidence in sorted(c_and_c, key=lambda x: x[1], reverse=True):
			show_box += (f'Vấn đề được dự đoán: {str(predicted_class)}, Khả năng chuẩn đoán: {round(confidence*100, 3)}%'+"\n")
		result.configure(text=show_box)
		if sub_class == 1:
			sub_class = "Nhẹ"
		elif sub_class == 2:
			sub_class = "Vừa"
		else:
			sub_class = "Nặng"
		result.configure(text=f'{show_box}\nMức nghiêm trọng được dự đoán: {sub_class}, Độ chính xác: {round(sub_confidence*100, 3)}%')
	else:
		pass

def predict_button_clicked():
	# smth ... lol :D
	predict()

print("successfully built all functions")

root.title("MID")
root.configure(bg='#2d3145')

label = tk.Label(root, text="Enter Text:", bg='#2d3145', fg='white')
label.pack()


label_status.pack()

entry = tk.Entry(root, bg='#35384a', fg='white')
entry.pack()

result = tk.Label(root, text="Kết Quả", bg='#2d3145', fg='white')
result.pack()

button = tk.Button(root, text="Predict", command=predict_button_clicked, bg='#272a3b', fg='white')
button.pack()

button = tk.Button(root, 
				   text="Record", 
				   command=recorder.record)
button.pack(side=tk.BOTTOM)

root.mainloop()