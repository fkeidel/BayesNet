# Water Sprinkler
Classical Bayesian Network example, see  
https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html  
and  
https://pyagrum.readthedocs.io/en/latest/notebooks/01-Tutorial.html  

- "Sprinkler" and "Rain" depend on "Cloudy":  
Cloudy weather makes it less likely that someone turns on his water sprinkler and makes it more likely that it will rain.
- "Wet Grass" depends on both, "Sprinkler" and "Rain".  
The grass is more likely to be wet if the water sprinkler was on or if it rained.  

The Bayesian Network encodes a joint probability distribution, which factorizes in:  
P(Cloudy,Sprinkler,Rain,Wet Grass) = P(Cloudy)\*P(Sprinkler|Cloudy)\*P(Rain|Cloudy)\*P(Wet Grass|Sprinkler,Rain)  
The tables in the diagram show the factors.

![](water_sprinkler.svg)