library(devtools)
devtools::install_github("nproellochs/ReinforcementLearning")
library("ReinforcementLearning")
#?gridworldEnvironment
set.seed()
env=gridworldEnvironment
states=c("s1","s2","s3","s4")
actions=c("up","down","left","right")
#?sampleExperience
data <- sampleExperience(N = 1000, env = env, states = states, actions = actions)
names(data)
head(data)
control=list(alpha=0.1, gamma=0.5, epsilon=0.1)
#?ReinforcementLearning
model <- ReinforcementLearning(data, s = "State", a = "Action", r = "Reward",s_new = "NextState", control = control)



