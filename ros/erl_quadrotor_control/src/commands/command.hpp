#ifndef ERL_COMMAND_HPP
#define ERL_COMMAND_HPP

namespace ERLControl{
    
    class PX4Quadcoptor;
    class Command{
        public:
        
        // sub-class this interface to create your own command for the quadcoptor
        // this class provided a pointer to the PX4Quadcoptor [quad] for use.

        // initialize is called only once when your command runs for the first time.
        virtual void initialize() {}

        // execute is called periodically until your [isFinished] function returns true
        virtual void execute() {}

        // return whether the command should finish.
        virtual bool isFinished() = 0;

        // called once when the command finishes.
        virtual void end(){}

        // override this message and change it to your command name.
        virtual std::string getName() { 
            return std::string("Base Command");
        }

        // main logic flow for commands. do not touch.
        void step(){
            if (process_ends){
                return;
            }

            if (!initialized){
                initialize();
                initialized = true;
            }

            if (!isFinished()){
                execute();
            }else{
                end();
                process_ends = true;
            }
        }

        bool isProcessEnded(){
            return process_ends;
        }

        void setPX4(PX4Quadcoptor* px4Ptr){
            quad = px4Ptr;
        }

        virtual void print_info(){
        }

        protected:
        PX4Quadcoptor* quad = nullptr;

        private:
        bool initialized = false;
        bool process_ends = false;
    };
}

#endif
