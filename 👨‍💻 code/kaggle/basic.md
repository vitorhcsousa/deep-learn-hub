Basic Intro on Kaggle Competition

## Understanding the data

1. Summarize the columns (type of data, meaning, different values)

| Column | dtype | Meaning | distinct values |
| ------ | ----- | ------- | --------------- |
| Col    | Int   | ….      | {}              |

- Check the sample 
-  We can use ```.info()``` command to help

2. Check the dataset details
   1. `.shape` returns the number of lines and colunms
   2. `.describe()`, applied only to numerical data gives some statistical information (min, max, count, 25%,50%,75%, avg, std)

## Analyzing the data

- Evaluating the shape of the dataset, the types of values, the number of null values, and the feature distribution, we will form a preliminary image of the dataset.
  - `missing_data()`
  - `most_frequent_value()`
  - `unique_values()`

### Univariante analysis

1. Set a unique color scheme for the notebook. Ensuring color and style unity across the entire notebook helps us to maintain the consistency of the presentation and ensures a well-balanced experience for the reader
2. Represent the distibutions of each feaures by dataset  and by the target
   1. `plot_count_pairs()` for categorical 
   2. `plot_distribution_pairs()` for continuous





## Feature Engineering 

