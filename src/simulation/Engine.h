//
// Created by test on 2022-02-09.
//

#ifndef MPM_SOLVER_ENGINE_H
#define MPM_SOLVER_ENGINE_H

#include "GridManager.h"
#include "ParticleManager.h"


namespace MPM {

	
	class Engine {



	public:
		void create();
		void step();
		




	private:
		void p2g();
		void updateGrid();
		void g2p();
		

		

		



	};

}



#endif //MPM_SOLVER_ENGINE_H
