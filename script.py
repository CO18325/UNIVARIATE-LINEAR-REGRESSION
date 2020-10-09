import matplotlib.pyplot as plt
plt.style.use('ggplot') 

''' %matplotlib inline
%matplotlib inline sets the backend of matplotlib to the 'inline' backend:
With this backend, the output of plotting commands is displayed inline within 
frontends like the Jupyter notebook, directly below the code cell that produced 
it. The resulting plots will then also be stored in the notebook document.
'''
import numpy as np 
import pandas as  pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D





# THIS FUNCTION DISPLAYS THE RAW DATA GIVEN TO US
# IN THE FORM OF A SCATTER PLOT
# THIS WILL HELP US TO UNDERSTAND THE DATA BETTER
def visualize_raw_data(data):
    
    # X-AXIS WILL CONTAIN THE POPULATION
    # Y-AXIS WILL CONTAIN THE PROFIT
    x = 'Population'
    y = 'Profit'
    title = 'POPULATION(in 10000) V/S PROFIT(in $10000)'
    # NOW WE CALL THE FUNCTION FROM SNS LIBRARY
    graph = sns.scatterplot(x=x,y=y,data=data)
    graph.set_title(title)

    # MATPLOTLIB FUNCTION TO SHOW ALL THE GRAPHS
    plt.show()






#FUNCTION COMPUTE COST
# X,y,y_pred ARE ALL MATRICES AND ALL OPERATIONS ARE MATRICE OPERATIONS
# EVEN THETA IS ALSO A MATRICE
#COST J(theta) IS PARAMETRIZED BY THETA MATRICE AND NOT X or y!!!
def cost_function(X,y,theta):
    
    # I.E. NO. OF ENTRES IN THE DATA SET
    m = len(y) 

    # THIS IS THE MATHEMATICAL DOT PRODUCT 
    # TO OBTAIN PREDICTION WRT TO THETA AND X
    y_pred = X.dot(theta)

    # FIND THE SQUARED ERROR
    sq_error = (y_pred - y) ** 2

    # AND THEN WE FINALLY RETURN THE COST FUNCTION CALCULATED 
    return 1/(m * 2) * np.sum(sq_error)





# GRADIENT DESCENT FUNCTION
# TO CALCULATE THE MINIMUM COST
# WE WILL USE AN ALGORITHM CALLED BATCH GRADIENT DESCENT
# WITH EACH ITERATION IN THIS ALGO THE PARAMETERS I.E. THETA COMES CLOSER TO THEIR OPTIMAL VALUE
def gradient_descent(X,y,theta,alpha,iterations):
    
    m = len(y)

    # WE WILL STORE THE COST FUNCTIONS CALCULATED EACH TIME HERE
    # THIS IS JUST TO VISUALIZE OUR CONVERGENCE TOWARDS OPTIMAL COST FUNCTION
    # WRT THE ITERATIONS DONE 
    costs = []

    for i in range(0,iterations):
        
        # CALCULATE THE PREDICTION WRT TO theta
        y_pred = X.dot(theta)

        # CALCULATE THE SUBMASSION PART OF THE GRADIENT DESENCT
        # GO BACK TO SLIDE
        error = np.dot(X.transpose(),(y_pred - y))

        # CHANGE THE THETA FOR ACHEIVING THE OPTIMAL COST FUNCTION
        theta -= alpha * error * (1/m)

        # STORE THE COST FUNCTION OBTAINED IN OUR costs VARIABLE
        # FOR FUTURE VISULAIZATIONS
        costs.append(cost_function(X,y,theta))


    # AND FINALLY RETURN THE OPTIMAL THETA FOUND
    # AND THE COSTS STORED OVER ITERATIONS        
    return theta,costs




# VISUALIZING THE COST FUNCTION ON A 3D GRAPH
# IN THIS FUNCTION WE ARE EXPLICITLY GIVING THETA VALUES TO THE COST FUNCTION
# FOR THESE THETA VALUES THE COST FUNCTION GIVES US THE COST
# WHICH IS STORED IN AN ARRAY (cost_values)
# THESE VALUES ARE NOW SENT TO THE graph_formation() FUNCTION
# IN graph_formation() THE ARRAY OF COST STORED IS USED TO CONSTRUCT A 3D GRAPH 
# THE PURPOSE OF VISUALIZATION IS TO SEE HOW GRADIENT FUNCTION WILL MOVE
def visualize_cost_function(X, y):
    
    # THE BELOW FUNCTION np.linespace DEFNES THAT
    # RETURN 100 EQUISACED VALUES B/W -10 AND 10
    theta_0 = np.linspace(-10,10,100)
    theta_1 = np.linspace(-1,4,100)

    # DEFINE A MATRICE TO STORE THETA VALUES
    #MATRICE OF SIZE THETA_0 X THETA_1
    # THIS IS DONE TO INCORPORATE EACH AND EVERY COMBINATION OF THETAS
    cost_values = np.zeros((len(theta_0),len(theta_1)))

    for i in range(len(theta_0)):
        for j in range(len(theta_1)):
            # GET CURRENT THETA VALUES
            specific_theta = np.array([theta_0[i], theta_1[j]])
             # CALCULATE THE COST BASED ON THE ABOVE DUMMY THETA VALUES
            cost_values[i,j] = cost_function(X, y, specific_theta)
    
    # NOW WE CAN FORM THE GRAPH
    graph_formation(theta_0,theta_1,cost_values)


def graph_formation(theta_0,theta_1,cost_values):
    

    fig = plt.figure(figsize=(12,8))
    # DEFINIG GRAPH TYPE
    graph = fig.gca(projection='3d')

    surf = graph.plot_surface(theta_0,theta_1,cost_values,cmap='viridis')
    # COLORBAR OR THE LEGEND
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('THETA_0')
    plt.ylabel('THETA_1')
    graph.set_zlabel('J OF THETA')

    # TO GIVE INITIAL ANGLE TO THE GRAPH FOR BETTER VIEW
    graph.view_init(30,330) 

    plt.show()





# PLOT THE CONVERGENCE
# IN THIS GRAPH WE ARE GOING TO PLOT J(THETA) AGAINST NO. OF ITERATIONS
# THE OBJECTIVE IS TO VIEW HOW J(THETA) IS MOVING TOWARDS MINIMUM VALUE IN EACH ITERATION
# THE Costs(i.e. J(THETA)) HAVE BEEN ALREADY CALCULATED USING gradient_descent FUNCTION
# ARGUMENTS : Cost AND No. of Iterations
def convergence_grah(costs, iterations):

    plt.plot(costs)
    plt.xlabel('ITERATIONS')
    plt.ylabel('J OF THETA')
    plt.title('COST FUNCTION VALUES VS ITERATIONS OF GRADIENT DESCENT')
    
    plt.show()





# THIS FUNCTION CONSTRUCTS TH BEST FIT LINEAR REGRESSION LINE ON OUR DATA
# IT USES THE FINAL VALUES OF THE THETA 
def regression_fit(data, theta):

    # FIRST WE WILL CONSTRUCT THE SCATTER PLOT WITH OUR DATA
    x = 'Population'
    y = 'Profit'
    title = 'REGRESSION FIT'
    graph = sns.scatterplot(x=x,y=y,data=data)
    graph.set_title(title)    

    # NOW WE WILL OVERLAY THE REGRESSION LINE
    #THETA IS A 2*1 ARRAY WHICH IS NOT FIT FOR MATRIC MULTIPLICATION
    # SO, WE NEED TO SQUEEZE THE THETA MATRICE
    theta = np.squeeze(theta) # NOW THETA IS 1*2 MATRICE OR WE CAN SAY AN ARRAY

    # NOW GETTING THE POINTS FOR THE LINEAR REGRESSION LINE
    x_value = [x for x in range(5,25)] # 5-25 AS OUR POPULATION IS BETWEEN THIS RANGE
    y_value = [(x*theta[1] + theta[0]) for x in x_value]  # PREDICTED VALUES FOR TRAINING DATA
    
    #SEABORN FUNCTION TO CONSTRUCT THE LINE
    sns.lineplot(x_value, y_value)

    plt.show()






# THIS FUNCTION WILL PREDCT THE PROFIT FOR UNKNOWN POPULATION
# USING THE FINAL THETA VALUES
# IT ALSO TAKES THE 2*1 X MATRICE AS ARGUMENT
# 2*1 MATRICE FOR SUCCESSFUL DOT PRODUCT WITH THETA
def predict_data(theta, X):

    theta = np.squeeze(theta)
    #print(theta.transpose())
    #print(X)
    y_pred = np.dot(theta.transpose(), X)
    return y_pred




#############################################################################




def main():
    
    # THIS IS THE SIZE OF THE WINDOW THAT WILL OPEN TO SHOW ALL THE GRAPHS
    plt.rcParams['figure.figsize'] = (12,8)


    # DATA RECEIVED
    # DATA RECEIVED IS M0STLY IN THE FORM OF DATA FRAME
    # DATA FRAME IS VERY USEFUL DATASTRUCTURE IN PANDAS LIBRARY IN TERMS OF DATA SET MANAGEMENT
    data = pd.read_csv('bike_sharing_data.txt')

    # TO PRINT FIRST FIVE ENTRIES OF THE DATASET TO UNDERSTAND THE DATA BETTER
    print(data.head())

    #POPULATION OF CITIES IN 10000s
    #PROFIT IN UNITS 10000 DOLLARS
    #TO GET MORE INFO ABOUT THE CSV FILE:-
    # THIS WILL HELP US UNDERSTAND THE DATA TYPES OF THE DATA
    # AND ALSO CHECK IF THERE IS ANY NULL VALUES PRESENT HERE
    print(data.info())

    #VISUALIZATION OF DATA 
    visualize_raw_data(data)


    #######################################################################################


    #SETTING UP REGRESSION VARIABLES:

    #I.E. NO. OF ENTRIES
    m = data.Population.values.size 

    #X is a Matrice of 1's and the Population Column
    # 1's is to accomodate the Theta0 i.e. the Interept
    # X is a m * 2 Matrice
    X = np.append(np.ones((m,1)), data.Population.values.reshape(m,1), axis=1)
    
    # y is the data set of the profits. It is also m*1 Matrice 
    y = data.Profit.values.reshape(m,1)
    
    #theta is a 2*1 Matrice
    #Initializing the values of theta with ZERO!
    # Although it is not a very good practice 
    theta = np.zeros((2,1))

    # JUST TO TEST THE COST FUNCTION
    print(cost_function(X, y, theta))

   
    #######################################################################################


    #LEARNING RATE
    alpha = 0.01

    #NO. OF ITERATION FOR CONVERGENCE
    iterations = 2000

    theta, costs = gradient_descent(X, y, theta, alpha, iterations)

    # TO CHECK THE VALUE OF h(x) AND FINAL VALUE OF THE THETA MATRICE
    print("h(x) = {} + {}x1".format(str(round(theta[0,0],2)), str(round(theta[1,0],2))))


    #VISUALIZE THE COST FUNCTION WITH EXPLICIT THETA VALUES IN A 3D GRAPH
    visualize_cost_function(X, y)

    # VISUALIZE THE COST WITH RESPECT TO NUMBER OF ITERATIONS OF GRADIENT DESCENT
    convergence_grah(costs, iterations)

    # TO VISUALIZE THE REGRESSION LINE
    regression_fit(data, theta)

    print(theta)

    # PREDICT THE RESULTS FOR UNKNOWN VALUES
    
    input_population = float(input('ENTER THE POPULATION IN 10000: '))
    
    # CONVERTING THE INPUT TO 2*1 MATRICE FOR DOT PRODUCT
    input_population_matrice = np.append( 1, input_population)
    #print(input_population_matrice)
    predicted_profit = predict_data(theta, input_population_matrice)
    print(predicted_profit)

    print('PROFIT FOR POPULATION OF {} WOULD BE Rs{}'.format(str(round(input_population*10000)), str(round(predicted_profit*1000))))



main()