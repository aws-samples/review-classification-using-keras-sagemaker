## Implementing a Review classification model with Keras, MXNet and SageMaker
The solution demonstrates a Distributed Learning system implementing Keras custom generator to train on sprase libsvm data.

#### Background
Data set: Amazon Customer Reviews Dataset
Amazon Customer Reviews (a.k.a. Product Reviews) is one of Amazon’s iconic products. In a period of over two decades since the first review in 1995, millions of Amazon customers have contributed over a hundred million reviews to express opinions and describe their experiences regarding products on the Amazon.com website. Over 130+ million customer reviews are available to researchers as part of this dataset.

#### Approach
1.	The Review classification is NLP machine learning model to predict whether a review posted by the customer is positive or negative. For the sake of simplification, we have converted the ratings provided by the customer into a binary target variable with value equals to 1 when ratings were either 4 or 5.

2.	In present work, after splitting the data into train and test we created a feature engineering pipeline on training data using natural language processing techniques. The dataset was further split into 10 chunks to demonstrate distributed learning, where during training not more than a single chunk is loaded on the memory at a time.

3.	The data was trained using SageMaker MXNet container in a script mode. In the present example the training script consisted of following function and their implementation.
    1.	f1 custom function to optmize around.
    2.	readfile function to read individual files
    3.	Custom generator to loop through files and records to return batchs in sequence
    4.	parse_args to parser to read all arguements
    5.	train function to train the Neural network model
    6.	threadsafe_iter class and threadsafe_generator to make custom generator threadsafe
    7.	input_fn function to convert inference payload to numpy vector.
    4.	The model was evaluated on the test data using SageMaker Batch Transform functionality.
5.	 Results
    1.	Model achieved an ROC-AUC of 0.97 on the holdout set below are some reviews classified as good and bad by the model
        Reviews classified as Good by the model
           1. 'Awesome show, wish they would have continued it!',
           2. 'Great movie.',
           3. 'Absolutely beautiful!!!',
           4. 'I love this show <3',
           5. 'One of my favorite kind of shows. Love I!'

          Reviews classified as Bad by the model
         1. If I could give this less than 1 star I would.  The worst movie I've seen in ages.  The acting was less than high school level.  The Mockingjay character was mind numbingly boring and droll.  She was terrible.  The characters were beyond pathetic.  1 hour and 40 minutes of pure torture and inane plot, script combined with terrible CGI and effects I found myself wanting to consume mass quantities of alcohol.  The whole Hunger Game sage is a total waste of time.  How anyone can say this is anything but pure boredom is beyond me.
         2. slow and boring, boring, boring, just like the book
         3. "Awful. A waste of money. A waste of time. Boring. Predictable. I gave up for the sake of my sanity about 20 minutes before it was over. Sandler and Barrymore must have been bored to tears with nothing to do, under contract to do something, and just didn't care........",
         4. 'Horrible movie. Mediocre acting, terrible plot, confusing and boring.',
         5. 'The movie is a real waste of time and money, poor actors, poor story, poor locations... really awful movie. I waste my money with this piece of crap.'

* Change the title in this README
* Edit your repository description on GitHub

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
