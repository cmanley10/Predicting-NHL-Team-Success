# Technical Report - Predicting NHL Playoff Success
## Problem Statement
The National Hockey League (NHL) has a reputation for being unpredictable. While there are many franchises that have enjoyed years of continued success and playoff berths, nothing is taken for granted in the playoffs. It is not uncommon for low seeded teams to win playoff series in the NHL. The 2012 Los Angeles Kings even won the Stanley Cup as an 8 seed. As a strict salary cap league, the NHL enjoys a level of parity that makes for an exceedingly exciting postseason.

With all that being said, my goal here is to see if it is possible to create a classification model that can predict playoff outcomes using regular season data from that year. The purpose for this model is twofold. First, with sports gambling becoming legalized and the increasing popularity of fantasy sports, an accurate predictive model would be very valuable to many people.

Second, I am aiming to keep the results of this model interpretable. I am interested to see which factors and statistics from the regular season are predictive of playoff success. If the model has a high degree of accuracy or predictive power, the results of those predictive factors and statistics would be valuable to NHL franchises in helping to build a "team profile for success."

#### Notes for Consideration
Trying to predict who will win the Stanley Cup, or any sporting championship for that matter, is a particularly difficult task. Many have tried with varying degrees of success, but there is no "magic model" out there that is performing head and shoulders above the rest (And if there was, there would be someone out there making a lot of money). This is the nature of predicting sports outcomes. I do not expect this model to be particularly accurate when predicting a champion. Rather, the true purpose is to see if I can engineer some features that will get me close. This project will be ongoing, and with more time I hope to improve the outcomes.   

  ---


## Gathering data
For this project, my primary source of data was [hockey reference](hockey.reference.com). All of the data was scraped from the hockey reference website using BeautifulSoup. I gathered 10 years of regular season team statistics from the 2008 season through the 2018 season. Additionally, I gathered basic and advanced statistics for each individual player over that same time span.

The datasets can be found on my personal [github](https://github.com/cmanley10/Predicting-NHL-Team-Success/tree/master/Data).

  ---

## Exploratory Data analysis
This project was extremely challenging in many ways. One of the most challenging aspects was getting the data into a format with which I could create a sensible model. Because I scraped my data, I ended up with many different csv files from which I constructed my dataframes. In order to predict team outcomes I knew that the data that I fed the model had to represent team statistics rather than individual player statistics. Due to this fact, I ended up aggregating individual player statistics into stats representative of the team as a whole.

To begin with, I created a 'rank' column in my dataframe which documented the final rank of every team for a given year. The Stanley Cup winner got a rank of 1 and the Stanley Cup runner up was ranked 2. The two conference finalists were ranked 3 and 4, with the higher ranking going to the team that finished with more points in the standings in the regular season. This same method was used for ranking the rest of the playoff teams. For non-playoff teams, they were simply ranked by their final regular season point totals.

Using this rank column I was able to begin to find correlations between team statistics and final rank. Here are some examples of interesting relationships I found:

**Average Plus Minus:**

 Plus Minus is an individual statistic that measure the goal differential while that player is on the ice. If your team scores while you are on the ice, your plus/minus increases by 1, and decreases by 1 if your team is scored against while you are on the ice. *Average Plus Minus* is the mean of every individual's plus minus on a given team. This is closely related to a team's goal differential.

  *(fig. 1)*

![Average Plus Minus](https://github.com/cmanley10/Predicting-NHL-Team-Success/blob/master/Visuals/Average%20Plus%20Minus%20Relative%20to%20Team%20Rank.png)

*Champions are shown in orange*

  ---

**Average Corsi For Percentage**

- Corsi is an individual player stat that calculates shots on goal + shots that are blocked + shot attempts that miss the goal. The stat is calculated as CF, or Corsi For and CA, or Corsi Against. A Corsi percentage is calculated for each player by taking CF/(CF + CA). A Corsi For Percentage over %50, means that the team was controlling the puck more often than not when that particular player is on the ice. To turn this into a team statistic I have taken the mean of the Corsi For Percentage for each player on a given team in a given year. The idea is we get a clearer view of the team's overall puck possession in that season.

*(fig. 2)*

![Average Corsi For Percentage](https://github.com/cmanley10/Predicting-NHL-Team-Success/blob/master/Visuals/Corsi%20Percentage%20Relative%20to%20Team%20Rank.png)

*Champions are shown in orange*

Many of the statistics I looked at have a strong correlation to 'rank', but not necessarily to the Stanley Cup Champion. This can be seen very clearly in **figure 1 - Average Plus Minus.**

  ---

### Feature Engineering
Going into this project I was looking for a way to quantify a teams depth. From my anecdotal experience with following the Stanley Cup Playoffs, it seems that the teams that are successful get production from many different players, rather than just one or two stars. To quantify a teams depth, I created the 'Balanced Scoring' statistic.

**The Balanced Scoring Function:**

This function takes in a dataframe of all skaters (excluding goaltenders) in a given year. The function returns a balanced scoring dataframe for that given year. Balanced scoring is calculated by taking the average of points scored for all players in that year. Then the standard deviation for points scored in that year is calculated and added to the mean. This finds all of the players who scored a number of points that is one standard deviation above the mean for that year. Essentially, these are a teams top scorers. Next, this calculates what percentage of players on a given team are a part of the 'top scorers' class. This percentage is found by dividing the number of top scorers on a given team by 18 (18 is the number of skaters that dress in a given game). Finally, the function returns a dataframe which includes the team, the balanced scoring calculation, and the year as columns. Let's take a look a closer look:

 *(fig. 3)*

 ![Balanced Scoring Function Code Snippet](https://github.com/cmanley10/Predicting-NHL-Team-Success/blob/master/Visuals/Balanced%20Scoring%20Function.png)

*(fig. 4)*

![Balanced Scoring 2018](https://github.com/cmanley10/Predicting-NHL-Team-Success/blob/master/Visuals/Balanced%20Scoring%20in%202018.png)

*Playoff teams in red*

 So, as we can see, in 2018 there is definitely a higher concentration of playoff teams who have balanced scoring. However, within those playoff teams, having more balanced scoring is not necessarily indicative of going further in the playoffs.

   ---

## Building a model
This project was challenging in many ways, but perhaps the most challenging aspect was determining what, in fact, I was predicting. Initially I had planned to predict only the Stanley Cup winner. This presented a big problem as there is only 1 cup winner in a given year out of 30 or 31 teams depending on the season. This is a huge class imbalance, coupled with the small number of observations (30/31) in a given year, creating a workable model from that data would be extremely difficult.

Instead I have chosen to assign a rank to each team in every season for which I have data. The teams are ranked based on where they finished. The specific method used for ranking is described in the Exploratory Data Analysis section. This rank value is dropped from the feature set and is the main target variable.

##### Multiclass Logistic regression
The Multiclass Logistic Regression will allow me to predict every teams final season ranking based on regular season statistics. In addition to receiving the numerical ranking for each team, I will be able to see the probabilities assigned to those predictions. While predicting how far each team will get in the playoffs is very difficult and very high accuracy is unlikely, assigning probabilities to those predictions is necessary for interpreting results.

 *- Notes on the train, test, split and overfitting:*
 Train, test, split is a little tricky due to the dataset. The purpose of the model is to predict playoff performance based on stats from the regular season. So, I cannot use an automated train, test, split here, as I need training data that contains all of the observations from a given year. Instead I have decided to manually select 8 whole years of data to use as my training data, while holding out 2 whole years to use as my testing data. I accept that the results of this model will be overfit and rather unreliable. Again, this is a big limitation with working with a dataset that has so few true observations.

  ---

## Results
As I suspected, the overall accuracy score is not very high. However, the model actually did better than I thought when considering how little data it had to work with.

**Accuracy Score on test data** - .18

While the score is very low, consider that the model has to make 30 predictions for every year. Furthermore, the teams are so tightly packed both in standings and team statistics. Let's take a look at an image of what my predictions look like:

*(fig. 5)*

![2016 Predictions](https://github.com/cmanley10/Predicting-NHL-Team-Success/blob/master/Visuals/2016%20Predicted%20Ranks%20and%20Probabilities.png)

So, as we can see, while the model does not do a great job of predicting the exact spot where a team will finish, it doesn't do a bad job of getting close. In this particular year it even accurately predicted the two finalists, Pittsburgh and San Jose. Although, it did not accurately predict the winner.

  ---

## Future Considerations
This was a fun project, and one I would consider building on. Having gone through the process of collecting this data and making predictions I can answer the initial question posed at the beginning of this project. Given regular season statistics, it is very difficult to build a classification model to predict postseason outcomes with high accuracy. Here is what I would do differently:

- Model Selection:
I now believe that a classification model may not be the best way to attempt to solve this problem. Because the model has to make so many predictions with so little data, it is very hesitant to make predictions that a team will be very highly ranked because there are so few examples of that in the data. This essentially boils down to a kind of unbalanced class problem. Additionally, the model as it is currently constructed, has no real understanding of 'other' teams, ie: head-to-head matchups.

- Next Idea:
What I would like to try next is to focus on the handful of statistics that are predictive of a single team beating another team in a given game. I need to focus more on specific match-ups rather than overall team statistics for the year. If I can assign a probability of Team A beating Team B in a playoff series, I can do that for all of the first round matchups. Given those probabilities I can find the conditional probabilities for a given team against all other teams they may face.  
