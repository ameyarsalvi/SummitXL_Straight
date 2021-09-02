classdef VrepEnvironment < rl.env.MATLABEnvironment
    
  
    % Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize the necessary properties of the environment     
        % Reward each time the car goes straight
        RewardForGoingStraight = 1
    
        % Penalty when the car does not goe straight
        PenaltyForTurning = -10
        
        Force
        
    end
    
    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State = zeros(6,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    % Necessary Methods (Step and Reward)
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = VrepEnvironment()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([6 1]);
            ObservationInfo.Name = 'Robot Velocities Observed';
            ObservationInfo.Description = 'vx, vy, vz, wx, wy, wz';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([4 1],'LowerLimit',[-0.1;-0.1;-0.1;-0.1],'UpperLimit',[0.1;0.1;0.1;0.1]);
            ActionInfo.Name = 'Wheel Velocities Given';
            ActionInfo.Description = 'w1, w2, w3, w4';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            %updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            % Get action
            this.Force = getForce(this,Action);
            
            sim = remApi('remoteApi');
            sim.simxFinish(-1);
            clientID = sim.simxStart('127.0.0.1', 19999, true, true, 5000, 10);
            if (clientID > -1)
                disp('Connected')

                %Handle
                [returnCode, LWB] = sim.simxGetObjectHandle(clientID, 'joint_back_left_wheel', sim.simx_opmode_blocking);
                [returnCode, RWB] = sim.simxGetObjectHandle(clientID, 'joint_back_right_wheel', sim.simx_opmode_blocking);
                [returnCode, LWF] = sim.simxGetObjectHandle(clientID, 'joint_front_left_wheel', sim.simx_opmode_blocking);
                [returnCode, RWF] = sim.simxGetObjectHandle(clientID, 'joint_front_right_wheel', sim.simx_opmode_blocking);
                [returnCode, SummitXL] = sim.simxGetObjectHandle(clientID, 'Summit_XL_visible', sim.simx_opmode_blocking);


                %First Call
                [returnCode,linearVel,angularVel]=sim.simxGetObjectVelocity(clientID, SummitXL, sim.simx_opmode_streaming);
                
                F=this.Force;
                
                [returnCode] = sim.simxSetJointTargetVelocity(clientID, RWB, F(1), sim.simx_opmode_blocking);       
                [returnCode] = sim.simxSetJointTargetVelocity(clientID, RWF, F(2), sim.simx_opmode_blocking);
                [returnCode] = sim.simxSetJointTargetVelocity(clientID, LWB,  F(3), sim.simx_opmode_blocking);
                [returnCode] = sim.simxSetJointTargetVelocity(clientID, LWF,  F(4), sim.simx_opmode_blocking);
                
                [returnCode,linearVel,angularVel]=sim.simxGetObjectVelocity(clientID, SummitXL, sim.simx_opmode_buffer);
                
                

                sim.simxFinish(-1);
            end

            sim.delete();
            
            this.State(1)=linearVel(1);
            this.State(2)=linearVel(2);
            this.State(3)=linearVel(3);
            this.State(4)=angularVel(1);
            this.State(5)=angularVel(2);
            this.State(6)=angularVel(3);
            
            vx=this.State(1);
            vy=this.State(2);
            vz=this.State(3);
            wx=this.State(4);
            wy=this.State(5);
            wz=this.State(6);
            
            % Euler integration
            Observation=zeros(6,1);
            Observation(1) = vx;
            Observation(2) = vy;
            Observation(3) = vz;
            Observation(4) = wx;
            Observation(5) = wy;
            Observation(6) = wz;
            

            % Update system states
            this.State = Observation;
            
            % Check terminal condition
            V_X = Observation(1);
            %rest=Observation(2:6);
            IsDone = V_X > 10;
            this.IsDone = IsDone;
            
            % Get reward
            Reward = getReward(this);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
         
            InitialObservation = [0;0;0;0;0;0];
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    % Optional Methods (set methods' attributes accordingly)
    methods               
        
        function force = getForce(~,action)
            force = action;           
        end
      
        % Reward function
        function Reward = getReward(this)
            Reward=(100*this.State(1)) - (this.State(2)^2 +this.State(3)^2 +this.State(1)^2 +this.State(2)^2 +this.State(3)^2);
        end
        
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
