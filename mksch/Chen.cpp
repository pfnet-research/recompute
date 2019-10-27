#include "dag_dp.h"
#include <time.h>
#include <list>
#include <algorithm>
#include <cmath>

typedef std::pair<std::pair<int,int>,std::vector<int>> ORDER;
typedef std::vector<ORDER> SCHEDULE;

const int INF = 1000000000;

static std::vector<int> nullvi;
static std::map<int,int> nullmii;
static std::map<int,std::pair<int,int>> nullmipii;

subgraph end;

int block_number[MAX_N] = {};
void show_blocks(const DAG &dag, std::string output = "blocks.dot");

int MEMORY_BUDGET = 20;
long long comp_ratio = 1;
const int MAX_TIME = 100000;
bool MAX_BATCHSIZE = false;
bool OPTIMIZER = true;

std::vector<FunctionNode> Functions_id;
void optimizer_(std::list<ORDER>& sch){
	std::list<ORDER>::iterator itr = sch.begin();
	while(itr != sch.end()){
		std::list<ORDER>::iterator itr_ = itr;
		++itr;
		if(itr_->fr.fr == 0)continue;
		while(itr_ != sch.begin()){
			std::list<ORDER>::iterator itr__ = itr_; --itr__;
			if(itr__->fr.fr == 0){
				bool b = false;
				for(int i = 0 ; i < Functions_id[itr__->fr.sc].iID.size() ; i ++){
					int ID = Functions_id[itr__->fr.sc].iID[i];
					std::string Ty = Functions_id[itr__->fr.sc].iType[i];
					if(Ty == "gradient")b |= (itr_->fr.sc == ID) && (itr_->sc[0] == 1);
					else b |= (itr_->fr.sc == ID) && (itr_->sc[0] == 0);
				}
				/*for(int i: Functions_id[itr__->fr.sc].iID){
					b |= itr_->fr.sc == i;
				}*/
				if(b)break;
				if(itr__->sc.size() == 0){
					b |= (itr_->fr.sc == Functions_id[itr__->fr.sc].oID[0]) && (itr_->sc[0] == 0);
				}
				else {
					for(int i: itr__->sc){
						b |= (itr_->fr.sc == Functions_id[itr__->fr.sc].oID[i]) && (itr_->sc[0] == 1);
					}
				}
				if(b){
					//erase
					break;
				}
			}
			sch.insert(itr__,*itr_);
			sch.erase(itr_--);
			--itr_;
		}
	}
}

SCHEDULE optimizer(SCHEDULE sch){
	std::list<ORDER> ret;
	for(ORDER ord: sch){
		ret.push_back(ord);
	}
	optimizer_(ret);
	sch.clear();
	for(ORDER ord: ret){
		sch.push_back(ord);
	}
	return sch;
}

long long eval(const DAG &dag, SCHEDULE sch){
	int ret = 0;
	subgraph competed_forward;
	subgraph competed_backward;
	for(ORDER ord: sch){
		if(ord.fr.fr == 0){
			FunctionNode func = Functions_id[ord.fr.sc];
			if(func.dir == "forward"){
				competed_forward[func.oID[0]] = true;
			}
			else {
				for(int ind: ord.sc){
					competed_backward[func.oID[ind]] = true;
				}
			}
			ret = std::max ( ret , dag.m_G(competed_forward) + dag.m_G(competed_backward) );
		}
		else {
			if(ord.sc[0] == 0)competed_forward[ord.fr.sc] = false;
			else competed_backward[ord.fr.sc] = false;
		}
	}
	return ret;
}

int main(int argc, char *argv[]){
	clock_t start_time = clock();
	if(!(argc&1)){
		puts("argc error");
		return 0;
	}
	else {
		for(int i = 1 ; 2*i < argc ; i ++){
			if(argv[2*i-1][1] == 'b'){
				long long MB = atoll(argv[2*i]);
				while(MB > 100000000){
					fprintf(stderr,"MEMORY_BUDGET is too large\n");
					MB /= 10;
					comp_ratio *= 10;
				}
				MEMORY_BUDGET = MB;
			}
			else if(argv[2*i-1][1] == 't') {
				if(argv[2*i][0] == 'M'){
					MAX_BATCHSIZE = true;
				}
			}
			else if((std::string)(argv[2*i-1]) == "-opt"){
				if((std::string)(argv[2*i]) == "off"){
					OPTIMIZER = false;
				}
			}
		}
	}
	
	static DAG forward;
	std::vector<FunctionNode> Functions;
	in_cg(forward,Functions);
	Functions_id = Functions;
	for(FunctionNode func: Functions){
		Functions_id[func.id] = func;
	}
	
	static DAG forward_;
	forward_.clear();
	std::vector<int> vec;
	std::vector<int> id_;
	for(int i = 0 ; i < forward.N ; i++){
		if(forward.pre[i].size() > 0){
			id_.push_back(vec.size());
			vec.push_back(i);
		}
		else id_.push_back(-1);
	}
	forward_.N = vec.size();
	for(int i = 0 ; i < forward_.N ; i ++){
		forward_.t[i] = forward.t[vec[i]];
		forward_.m[i] = forward.m[vec[i]];
		forward_.p[i] = forward.p[vec[i]];
	}
	for(int i = 0 ; i < forward.N ; i ++){
		if(id_[i] == -1)continue;
		for(int j: forward.nex[i]){
			if(id_[j] != -1){
				forward_.add_edge(id_[i],id_[j]);
			}
		}
	}
	std::swap(forward,forward_);
	for(int i = 0 ; i < forward.N ; i ++)end[i] = true;
	show(forward,"forward_without_input.dot");
	
	long long memory_naive = 0;
	for(int i = 0 ; i < forward.N ; i ++){
		memory_naive += forward.m[i];
	}
	memory_naive /= comp_ratio;
	while(memory_naive > 100000000){
		fprintf(stderr,"sum of memory sizes is too large\n");
		memory_naive /= 10;
		MEMORY_BUDGET /= 10;
		comp_ratio *= 10;
	}
	for(int i = 0 ; i < forward.N ; i ++){
		forward.m[i] /= comp_ratio;
	}
	
	forward.make_cut_graph();
	/*static std::vector<std::vector<int>> dp_to;
	static std::vector<std::vector<int>> dt;
	static std::vector<std::vector<int>> dm;
	static std::vector<std::vector<int>> peak;
	static std::vector<int> cut_mem;
	static std::vector<int> peak_back;
	for(int j = 0 ; j < forward.cuts.size() ; j ++){
		cut_mem.push_back(forward.m_G(forward.cuts[j]));
		subgraph A = forward.delpl(forward.cuts[j])&(end^forward.cuts[j]);
		subgraph B = forward.delmi(A)&(end^forward.cuts[j]);
		peak_back.push_back(forward.m_G(A)+forward.m_G(B));
	}
	for(int i = 0 ; i < forward.cuts.size() ; i ++){
		dp_to.push_back(nullvi);
		dt.push_back(nullvi);
		dm.push_back(nullvi);
		peak.push_back(nullvi);
		for(int j = i+1 ; j < forward.cuts.size() ; j ++){
			if((forward.cuts[j]&forward.cuts[i]) != forward.cuts[i]){
				continue;
			}
			subgraph W = forward.delmi(end^forward.cuts[j])&(forward.cuts[j]^forward.cuts[i]);
			dp_to[i].push_back(j);
			dt[i].push_back(forward.t_G(W));
			dm[i].push_back(forward.m_G(W));
			peak[i].push_back(2*(cut_mem[j]-cut_mem[i])+peak_back[j]);
		}
	}
	
	for(int i = 0 ; i < forward.cuts.size() ; i ++){
		std::vector<std::pair<int,int>> vs;
		for(int j = 0 ; j < peak[i].size() ; j ++){
			vs.push_back(std::pair<int,int>(peak[i][j],j));
		}
		std::sort(vs.begin(),vs.end());
		std::vector<int> dp_to_, dt_, dm_, peak_;
		for(int j = 0 ; j < vs.size() ; j ++){
			int id = vs[j].sc;
			dp_to_.push_back(dp_to[i][id]);
			dt_.push_back(dt[i][id]);
			dm_.push_back(dm[i][id]);
			peak_.push_back(peak[i][id]);
		}
		dp_to[i] = dp_to_;
		dt[i] = dt_;
		dm[i] = dm_;
		peak[i] = peak_;
	}
	
	clock_t mid_time = clock();
	fprintf(stderr,"before culc. finished: %ld sec\n",(mid_time-start_time)/CLOCKS_PER_SEC);*/
	
	/*static std::vector<std::map<int,int>> dp;
	static std::vector<std::map<int,std::pair<int,int>>> dp_pre;
	for(int i = 0 ; i < forward.cut_id.size() ; i ++){
		dp.push_back(nullmii);
		dp_pre.push_back(nullmipii);
	}
	dp[0][0] = 0;
	for(int i = 0 ; i < forward.cut_id.size() ; i ++){
		for(std::pair<int,int> p: dp[i]){
			std::pair<int,int> pre = std::pair<int,int>(i,p.fr);
			for(int j = 0 ; j < dp_to[i].size() ; j ++){
				int to = dp_to[i][j];
				if(p.sc+peak[i][j] <= MEMORY_BUDGET){
					int newtime = p.fr + dt[i][j];
					int newcost = p.sc + dm[i][j];
					std::map<int,int>::iterator itr = dp[to].lower_bound(newtime);
					if(itr != dp[to].end() && itr->sc <= newcost)continue;
					if(itr != dp[to].begin()){
						if(itr == dp[to].end() || itr->fr != newtime)--itr;
						while(itr->sc >= newcost){
							if(itr == dp[to].begin()){
								dp[to].erase(itr);
								break;
							}
							else dp[to].erase(itr--);
						}
					}
					dp[to][newtime] = newcost;
					dp_pre[to][newtime] = pre;
				}
				else break;
			}
		}
	}
	
	clock_t end_time = clock();
	fprintf(stderr,"DP finished: %ld sec\n",(end_time-start_time)/CLOCKS_PER_SEC);*/
	
	std::vector<int> vs;
	/*int end_id = forward.cut_id[end];
	std::pair<int,int> loc = std::pair<int,int>(0,0);
	if(dp[end_id].size() == 0){
		std::cerr << "IMPOSSIBLE" << std::endl;
		puts("IMPOSSIBLE");
		return 0;
	}
	if(MAX_BATCHSIZE){
		loc = *dp[end_id].begin();
		fprintf(stderr,"max batchsize mode\n");
	}
	else {
		loc = *--dp[end_id].end();
	}
	loc.sc = loc.fr;
	loc.fr = end_id;
	subgraph checkpoints;
	int block_cnt = 0;
	while(1){
		vs.push_back(loc.fr);
		if(loc.fr == 0)break;
		loc = dp_pre[loc.fr][loc.sc];
	}*/
	
	//Chen
	std::vector<std::pair<int,int>> ArtNodes;
	for(int i = 0 ; i < forward.cuts.size() ; i ++){
		subgraph s = forward.cuts[i];
		subgraph t = forward.delmi(end^s)&s;
		if(t.count() != 1 && s != end && s.any())continue;
		ArtNodes.push_back(std::pair<int,int>(forward.m_G(s),i));
	}
	std::sort(ArtNodes.begin(),ArtNodes.end());
	//std::reverse(ArtNodes.begin(),ArtNodes.end());
	/*for(std::pair<int,int> p: ArtNodes){
		vs.push_back(p.sc);
	}*/
	int hoge = std::max( std::sqrt((long double)ArtNodes.size()) , (long double)1.0 );
	int mem_sum = forward.m_G(end);
	int hoge_ = mem_sum/hoge;
	//std::cerr << ArtNodes.size() << std::endl;
	//std::cerr << hoge << std::endl;
	for(int i = 0 ; i < ArtNodes.size() ; i ++){
		if(i == 0 || i+1 == ArtNodes.size() || ArtNodes[i].fr > hoge_){
			vs.push_back(ArtNodes[i].sc);
			while(hoge_ < ArtNodes[i].fr)hoge_ += mem_sum/hoge;
		}
	}
	std::reverse(vs.begin(),vs.end());
	std::cerr << hoge << " " << vs.size() << std::endl;
	
	std::vector<subgraph> groups;
	std::vector<subgraph> W;
	std::vector<subgraph> A;
	std::vector<subgraph> B;
	for(int i = 0 ; i < vs.size() ; i ++){
		groups.push_back(forward.cuts[vs[i]]);
		subgraph w = forward.delmi(end^forward.cuts[vs[i]])&forward.cuts[vs[i]];
		W.push_back(w);
	}
	/*for(int i = 0 ; i < groups.size() ; i ++){
		std::cout << groups[i] << std::endl;
	}*/
	for(int i = 0 ; i+1 < vs.size() ; i ++){
		groups[i] ^= groups[i+1];
		subgraph a = forward.delpl(forward.cuts[vs[i+1]])&(end^forward.cuts[vs[i+1]]);
		subgraph b = forward.delmi(a)&(end^forward.cuts[vs[i+1]]);
		A.push_back(a);
		B.push_back(b);
	}
	for(int i = -2+(int)vs.size() ; i >= 0 ; i --){
		W[i] |= W[i+1];
	}
	groups.pop_back();
	W.pop_back();
	std::reverse(groups.begin(),groups.end());
	std::reverse(W.begin(),W.end());
	std::reverse(A.begin(),A.end());
	std::reverse(B.begin(),B.end());
	subgraph computed_forward;
	subgraph computed_backward;
	std::vector<int> computed_func(Functions.size());
	/*for(int i = 0 ; i < groups.size() ; i ++){
		std::cout << i << ":\n";
		std::cout << groups[i] << std::endl;
		std::cout << W[i] << std::endl;
		std::cout << A[i] << std::endl;
		std::cout << B[i] << std::endl;
		puts("---------------");
	}
	for(int i = 0 ; i < forward_.N ; i ++){
		printf("%d ",id_[i]);
	}
	printf("\n");
	for(int i: vec){
		printf("%d ",i);
	}
	printf("\n");*/
	
	SCHEDULE sch;
	ORDER ord;
	
	for(int i = 0 ; i < groups.size() ; i ++){
		subgraph s = groups[i];
		for(FunctionNode func: Functions){
			if(func.dir != "forward")continue;
			for(int b: func.oID){
				if(id_[b] != -1 && s[id_[b]] && !computed_forward[b]){
					//printf("compute %d\n",func.id);
					ord.fr.fr = 0;
					ord.fr.sc = func.id;
					ord.sc.clear();
					sch.push_back(ord);
					computed_forward[b] = true;
				}
			}
		}
		for(int j = 0 ; j < forward.N ; j ++){
			if(computed_forward[vec[j]] && !W[i][j]){
				//printf("forget %d forward\n",vec[j]);
				ord.fr.fr = 1;
				ord.fr.sc = vec[j];
				ord.sc.clear();
				ord.sc.push_back(0);
				sch.push_back(ord);
				computed_forward[vec[j]] = false;
			}
		}
	}
	for(int i = groups.size()-1 ; i >= 0 ; i --){
		subgraph s = groups[i];
		for(FunctionNode func: Functions){
			if(func.dir != "forward")continue;
			for(int b: func.oID){
				if(id_[b] != -1 && s[id_[b]] && !computed_forward[b]){
					//printf("compute %d\n",func.id);
					ord.fr.fr = 0;
					ord.fr.sc = func.id;
					ord.sc.clear();
					sch.push_back(ord);
					computed_forward[b] = true;
				}
			}
		}
		for(int func_ = Functions.size()-1 ; func_ >= 0 ; func_ --){
			FunctionNode func = Functions[func_];
			if(func.dir != "backward")continue;
			std::vector<int> index;
			bool run = false;
			for(int j = 0 ; j < func.oID.size() ; j ++){
				int b = func.oID[j];
				if(id_[b] == -1){
					if(!computed_func[func.id])index.push_back(j);
				}
				else {
					if(s[id_[b]]){
						index.push_back(j);
						run = true;
					}
				}
			}
			//kowaii
			int out;
			for(int j = 0 ; j < func.iID.size() ; j ++){
				if(func.iType[j] == "gradient"){
					out = func.iID[j];
					break;
				}
			}
			//if(!run)run |= index.size() == func.oID.size() && s[id_[Functions[func.id-Functions.size()/2].oID[0]]];
			if(!run)run |= index.size() == func.oID.size() && s[id_[out]];
			if(!run)continue;
			
			//ReLU Exception1
			bool ReLUException = !computed_forward[out];
			if(ReLUException){
				ReLUException = false;
				for(std::string s: func.iType){
					if(s == "output")ReLUException = true;
				}
			}
			if(ReLUException){
				//if(func.FunctionType == "ReLU"){
					//printf("compute %d\n",func.id-(int)Functions.size()/2);
					ord.fr.fr = 0;
					ord.fr.sc = func.id-(int)Functions.size()/2;
					ord.sc.clear();
					sch.push_back(ord);
				//}
			}
			
			//printf("compute %d",func.id);
			ord.fr.fr = 0;
			ord.fr.sc = func.id;
			ord.sc.clear();
			for(int ind: index){
				//printf(" %d",ind);
				ord.sc.push_back(ind);
			}
			//printf("\n");
			sch.push_back(ord);
			
			//ReLU Exception2
			if(ReLUException){
				//if(func.FunctionType == "ReLU"){
					//printf("forget %d forward\n",Functions[func.id-Functions.size()/2].oID[0]);
					//printf("forget %d forward\n",out);
					ord.fr.fr = 1;
					ord.fr.sc = out;
					ord.sc.clear();
					ord.sc.push_back(0);
					sch.push_back(ord);
				//}
			}
			
			computed_func[func.id] = true;
		}
		for(int j = 0 ; j < forward.N ; j ++){
			if(s[j])computed_backward[vec[j]] = true;
		}
		for(int j = 0 ; j < forward.N ; j ++){
			//ayasii
			if(!forward.cuts[vs[vs.size()-1-i]][j] && computed_forward[vec[j]] && !B[i][j]){
				//printf("forget %d forward\n",vec[j]);
				ord.fr.fr = 1;
				ord.fr.sc = vec[j];
				ord.sc.clear();
				ord.sc.push_back(0);
				sch.push_back(ord);
				computed_forward[vec[j]] = false;
			}
		}
		for(int j = 0 ; j < forward.N ; j ++){
			if(computed_backward[vec[j]] && !A[i][j]){
				//printf("forget %d backward\n",vec[j]);
				ord.fr.fr = 1;
				ord.fr.sc = vec[j];
				ord.sc.clear();
				ord.sc.push_back(1);
				sch.push_back(ord);
				computed_backward[vec[j]] = false;
			}
		}
	}
	
	//optimize
	if(OPTIMIZER){
		std::cerr << "OPTIMIZER: on" << std::endl;
		sch = optimizer(sch);
	}
	else {
		std::cerr << "OPTIMIZER: off" << std::endl;
	}
	
	//output schedule
	for(ORDER ord_: sch){
		if(ord_.fr.fr == 0){
			printf("compute %d",ord_.fr.sc);
			for(int ind: ord_.sc){
				printf(" %d",ind);
			}
			printf("\n");
		}
		else{
			printf("forget %d ",ord_.fr.sc);
			if(ord_.sc[0] == 0)printf("forward\n");
			else printf("backward\n");
		}
	}
	
	//visualizer
	for(int i = 0 ; i < groups.size() ; i ++){
		subgraph s = groups[i];
		for(int j = 0 ; j < forward.N ; j ++){
			if(s[j])block_number[j] = i;
		}
	}
	show_blocks(forward,"blocks.dot");
	
	//
	fprintf(stderr,"#V: %d\n",forward.N);
	fprintf(stderr,"#L: %d\n",(int)forward.cuts.size());
	long long baseline_memory = 0;
	for(int i = 0 ; i < forward_.N ; i ++){
		baseline_memory += 2*forward_.m[i];
	}
	long long prop_memory = eval(forward_,sch);
	fprintf(stderr,"naive: %lld\n",baseline_memory);
	fprintf(stderr,"prop.: %lld\n",prop_memory);
	fprintf(stderr,"%lf \%\n", (double)prop_memory*100.0/(double)baseline_memory);
	
	return 0;
}

void show_blocks(const DAG &dag, std::string output){
	FILE* out = fopen( output.c_str() , "w" );
	fprintf(out,"digraph G{\n");
	fprintf(out,"clusterrank=local;\n");
	fprintf(out,"rankdir=LR;\n");
	for(int i = 0 ; i < dag.N ; i ++){
		if(dag.p[i] >= 2)continue;
		char fillcolor[20] = "#000000";
		int bnum = block_number[i];
		for(int i = 0 ; i < 3 ; i ++){
			if(bnum%3 >= 1){
				fillcolor[2*i+1] = (bnum%3==1)?'7':'F';
				fillcolor[2*i+2] = 'F';
			}
			bnum /= 3;
		}
		fprintf(out,"%d[label = %d, style = \"filled\", fillcolor = \"%s\"];",i,dag.m[i],fillcolor);
		for(int j: dag.nex[i]){
			if(dag.p[j] >= 2)continue;
			fprintf(out,"%d -> %d;",i,j);
		}
		/*if(dag.nex_copy[i] != i){
			fprintf(out,"%d -> %d[style=dotted,arrowhead=none];",i,dag.nex_copy[i]);
			fprintf(out,"{rank = same; %d; %d;}",i,dag.nex_copy[i]);
		}*/
	}
	fprintf(out,"\n}\n");
	fclose(out);
}
