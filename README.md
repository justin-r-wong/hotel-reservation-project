# Prediction on Hotel Reservation Cancellations

### Background and the Problem

With the rise of online hotel reservation channels, customer and hotel booking behaviors have been radically changed by various factors. The factors, such as, scheduling conflicts, changes in plans, no-shows, or even booking prices can cause such reservation cancellations. With many unexpected factors coming from customers that are difficult for hotel owners to identify and react to it before hand, the aim for this project is help hotel owners to better understand if the customer will keep the reservation or not based on a given set of features.

### Data Description

The data provided comes in a tabular form. Since we are performing a machine learning prediction based on the dataset, we first work on our prediction models on the training dataset for verification purposes, and then we perform the actual prediction on the testing dataset to see how well our models were trained on. Our training data consists of 12695 examples (ie. rows) with 18 columns. The list below identifies the columns that were provided in the dataset (sourced from the dataset provider: https://www.kaggle.com/datasets/gauravduttakiit/reservation-cancellation-prediction?select=train__dataset.csv):

• Booking_ID: unique identifier of each booking

• No of adults: Number of adults

• No of children: Number of Children

• noofweekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel

• noofweek_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel

• typeofmeal_plan: Type of meal plan booked by the customer

• requiredcarparking_space: Does the customer require a car parking space? (0 - No, 1- Yes)

• roomtypereserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.

• lead_time: Number of days between the date of booking and the arrival date

• arrival_year: Year of arrival date

• arrival_month: Month of arrival date

• arrival_date: Date of the month

• Market segment type: Market segment designation.

• repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)

• noofprevious_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking

• noofpreviousbookingsnot_canceled: Number of previous bookings not canceled by the customer prior to the current booking

• avgpriceper_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)

• noofspecial_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)

• booking_status: Flag indicating if the booking was canceled or not.

We found that no missing values were identified in each cell. Additionally, booking status is our target column, in which we want to make predictions on. A flag of 0 (class 0) indicates that the booking is not cancelled and a flag of 1 (class 1) indicates that the booking is cancelled.

### EDA

Our preliminary EDA shows some insights about the dataset and potential relationships that can capture the nature of the predictions. For instance, we found that one of the features in the dataset, lead time (ie. the number of days between between booking time and arrival time) can be used as a single indicator to perform the prediction. From the histogram of lead times below, we can say that if the lead time is greater than 150, then we predict class 1.

<img width="628" alt="lead_time" src="https://user-images.githubusercontent.com/52024770/231291281-8d9e58ec-2240-42bc-a60c-a11b305f6383.png">

*With regards to the numerical features for our dataset, we plotted a correlation matrix. The higher the magnitude of the correlation matrix between the two variables, the strongly the two variables are related to each other. We see that the highest positive correlation occur between number of previous bookings not canceled and number of previous booking cancelled, which suggests some linear relationship in between the two features contributing to the prediction result. Overall, a lot of the features have relative low correlation in terms of magnitude.*

<img width="893" alt="corr_matx" src="https://user-images.githubusercontent.com/52024770/231291346-9294e467-e37f-44a2-8915-c377e74eb7d7.png">

Other than the visualizations, we also obtained the proportion of each class in the target column. Class 1 (cancelled booking) takes about 33% of the training data and the remaining 67% being class 0 (not cancelled booking). This means that there is some class imbalance, with cancelled class as the minority class and not-cancelled class as the majority class. This may be something that needs to be taken into account when training models, as it could lead to biases in predictions.

### Model Construction and Feature Importance

To begin fitting our model, we fitted a baseline Dummy Classifier. It produced a recall of about 0.33. We selected recall because we are interested in finding out customers that are likely to cancel their booking status. In the context of recall, class 1 is considered to be the class of interest (ie. the positive class). Next, we fitted a logistic regression model with tuned hyperparameter C. With C = 10, we obtained a recall of 0.65. This indicates that the logistic regression model performs relatively better than the baseline model, however it does not fit the model extremely well.

Next, we fitted more complex models to our data. We chose to use SVM, Random Forest, XGBoost, and LGBM. As expected, with default parameters, the models we selected (SVM, RF, XGBoost, LGBM) all performed way better than logistic regression. The best model comes from XGBoost, with a recall score of 0.782. Random forest and LGBM have close enough recall scores and SVM beats logistic regression in recall score by 4%.

Since, the recall for random forest and the other ensemble methods weren't that far apart, we decided to tune hyperparamters from random forest classifier with the given computational resources. Via Random Search, we obtained the best recall score on the validation set with maximum tree depth of 20 and 666 estimators (ie. the number of trees used). We obtained a recall of 0.783, which is an improvement from random forest with default hyperparameters.

Besides, we also obtained the top 20 most important features in our random forest classifier model. The table below shows these features and their corresponding weights with the confidence interval. Some of these important features are lead time, average price per room, number of special requests, and number of week nights. We can see that all these features have positive weights in the model, which means that as these feature increase in value, the probability for predicting to cancel the hotel booking will increase as well.

<img width="300" alt="features" src="https://user-images.githubusercontent.com/52024770/231291451-765df9a2-9da0-492d-9017-e11abc9ab5c3.jpg">


### Results and Discussion

Lastly, we used our tuned random forest classifier on the test set. We obtained a test recall score of 0.796, which was just higher than the validation recall score by around 1.3%. Since the validation score isn't way higher than the testing score, we did not attribute this difference to optimization bias. The performance of the model during cross validation was comparable to running on the whole test set.

The plot below shows the ROC curve, which shows how well a binary classification model is able to discriminate between two classes, denoted as class 0 (negative class) and class 1 (positive class). From the plot, we were able to obtain a relatively high recall score for our test set but also obtained a false positive rate, which is ideally something we want to achieve, since we hope our model is able to classify class 1 as much as possible but at the same time to be correct.

<img width="626" alt="roc" src="https://user-images.githubusercontent.com/52024770/231291525-5b93ff3e-114e-4ceb-a1c7-26317baf2bb8.png">

Evaluating our results it could be a possibility that we may be overfitting our model. Since the training recall is quite high for random forest, we should investigate on the training and validation score graph and observe the sweet spot for our hyperparameters. We fit complete random forest and let it fully grow to maximum depth, so it is likely overfitting could happen there. We are not sure whether optimization bias really happened or if it is just due to luck that the test set and the training set looks similar. It is shown that the validation score and the test score are pretty close and we concluded that the validation score is representative of the testing score. If we have access to a larger data set than this, perhaps, this could be something we would look into.

As for further improvements, we could try to perform hyperparameter tunings and other models (ie. LogisticRegression, XGBoost, SVC, etc...) and compare the scores with our current random forest model. Because we see some sort of class imbalance in this dataset, we could potentially try to adjust the class weights in the models as well. We could also try to check the performance of the model with just including the most important features from various feature selection methods and see if the model performance is similar or better with just those features on the test set.
