
clc
clear all
env=VrepEnvironment();
validateEnvironment(env);


observationInfo = getObservationInfo(env);
numObservations = observationInfo.Dimension(1);
actionInfo = getActionInfo(env);
numActions = actionInfo.Dimension(1);

%%


L = 60; % number of neurons
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(100,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(100,'Name','fc4')
    reluLayer('Name','relu4')
    fullyConnectedLayer(1,'Name','fc9')];

actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(L,'Name','fc10')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
    
criticNetwork = connectLayers(criticNetwork,'fc10','add/in2');


%%

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);

critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);

%%

actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(100,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(100,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(75,'Name','fc4')
    reluLayer('Name','relu4')
    fullyConnectedLayer(numActions,'Name','fc8')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',[0.1;0.1;0.1;0.1],'Bias',[0;0;0;0])];

%%

actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);

%%

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',0.02,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',10000,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64);

agentOptions.NoiseOptions.Variance = [1;1;1;1];
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;
agentOptions.ResetExperienceBufferBeforeTraining = true;
%%
%agentOptions.ResetExperienceBufferBeforeTraining = false;

agent = rlDDPGAgent(actor,critic,agentOptions);
%%
maxepisodes = 100;
maxsteps = 10;
trainingOpts = rlTrainingOptions('MaxEpisodes',maxepisodes,'MaxStepsPerEpisode',maxsteps,'Verbose',true,'StopTrainingCriteria','GlobalStepCount','StopTrainingValue',125000,'Plots',"training-progress");



%%
trainingStats = train(agent,env,trainingOpts);

%%

save("StraightLineAgent.mat",'agent')

%%
%load('StraightLineAgent.mat')

%simOpts = rlSimulationOptions('MaxSteps',1000);
%experience = sim(env,agent,simOpts);
