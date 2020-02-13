from flask import Flask, render_template, url_for, request, redirect
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/video")
def videos():
    return render_template('video.html')

@app.route("/image")
def images():
    return render_template('images.html')

@app.route("/news")
def news():
    return render_template('news.html')

@app.route('/upload-text-simple', methods=['GET', 'POST'])
def upload_text_simple():
    if request.method == 'POST':

        #-----Whatever happens here to find if news is fake or real-----

        # textfeed = request.form['textfeed']
        # model = joblib.load('./FN/newsmodel.pkl') 
        # tfidf_vect = joblib.load('./FN/vectorizer.pickle')
        # mylist = []
        # mylist.append(textfeed)
        # mylist = list(mylist)
        # df2 = pd.DataFrame(mylist, columns = ['textinput'])
        # myX = df2.textinput
        # mytest = tfidf_vect.transform(myX)
        # #return mytest
        # prediction = str(model.predict(mytest)[0])

        prediction = '1'

        print('pred: ', prediction)
        if prediction == '1':
            prediction = "It's Real"
        else:
            prediction = " It's Fake"
        return render_template('news.html', prediction=prediction)
    else:
        return redirect(url_for('news'))


@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST' and 'photo' in request.files:

        #---------Do the magic stuff and find if video is REAL or FAKE---------------

        # filename = photos.save(request.files['photo'])
        # classifier.load('weights/Meso4_F2F')

        # predictions = compute_accuracy(classifier, 'test_videos')

        # for video_name in predictions:
        #     answer=str(predictions[video_name][0])
        #     file1 = open("myvideo.txt","w") 
        #     #L = [answer]  
        #     file1.write(answer)
        #     file1.close() 
        #     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
        # return filename

        prediction = "It's Real"  # Placeholder for now
        return render_template('video.html', prediction=prediction)
    else:
        return redirect(url_for('videos'))


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    #-------- include this tb thing also------------
    #tb._SYMBOLIC_SCOPE.value = True
    if request.method == 'POST' and 'photo' in request.files:
        # filename = photos.save(request.files['photo'])
        # classifier = Meso4()
        # classifier.load('weights/Meso4_DF')

        # # 2 - Minimial image generator
        # # We did use it to read and compute the prediction by batchs on test videos
        # # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

        # dataGenerator = ImageDataGenerator(rescale=1./255)
        # generator = dataGenerator.flow_from_directory('test_images',target_size=(256, 256), batch_size=1,class_mode='binary',subset='training')

        # # 3 - Predict
        # X, y = generator.next()
        # answer=classifier.predict(X)
        # print('Predicted :', answer)
        # answer=str(answer)
        # file1 = open("myfile.txt","w") 
        # L = ["This is Delhi \n","This is Paris \n","This is London \n"]  
        # file1.close() 
        # print("write file")
        # # if(image_label>0.8):
        # #     image_label=1
        
        # # else:
        # #     image_label=0
        # print('function executed')
        # os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        # return filename

        prediction = "Real"
        return render_template('images.html', prediction=prediction)

    else:
        return redirect(url_for('images'))


if __name__ == '__main__':
    app.run(debug=True)