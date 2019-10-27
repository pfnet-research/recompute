#include "dag_dp.h"

static std::vector<int> nullvi;

void DAG::clear(int _N){
	N = _N;
	for(int i = 0 ; i < MAX_N ; i ++){
		nex[i].clear();
		pre[i].clear();
	}
}

void DAG::add_edge(int a,int b){
	nex[a].push_back(b);
	pre[b].push_back(a);
}

void DAG::make_cut_graph(){
	subgraph start;
	/*for(int i = 0 ; i < N ; i ++){
		start[i] = pre[i].size() == 0;
	}*/
	cuts.push_back(start);
	int siz = cut_id.size();
	cut_id[start] = siz;
	cut_nex.push_back(nullvi);
	std::stack<int> sta;
	sta.push(0);
	while(!sta.empty()){
		auto id_s = sta.top(); sta.pop();
		subgraph s = cuts[id_s];
		for(int i = 0 ; i < N ; i ++){
			if(s[i])continue;
			bool b = true;
			for(int j: pre[i]){
				if(!s[j]){
					b = false;
					break;
				}
			}
			if(!b)continue;
			s[i] = true;
			auto p = cut_id.emplace(s,cut_id.size());
			if(p.sc){
				cuts.push_back(s);
				sta.push(cut_id.size()-1);
				cut_nex.push_back(nullvi);
			}
			cut_nex[id_s].push_back(p.fr->sc);
			s[i] = false;
		}
	}
}

subgraph DAG::delpl(const subgraph &s) const {
	subgraph ret;
	for(int i = 0 ; i < s.size() ; i ++){
		if(!s[i])continue;
		for(int j: nex[i]){
			ret[j] = true;
		}
	}
	return ret;
}
subgraph DAG::delmi(const subgraph &s) const {
	subgraph ret;
	for(int i = 0 ; i < s.size() ; i ++){
		if(!s[i])continue;
		for(int j: pre[i]){
			ret[j] = true;
		}
	}
	return ret;
}
int DAG::m_G(const subgraph &s) const{
	int ret = 0;
	for(int i = 0 ; i < N ; i ++){
		if(s[i])ret += m[i];
	}
	return ret;
}
int DAG::t_G(const subgraph &s) const{
	int ret = 0;
	for(int i = 0 ; i < N ; i ++){
		if(s[i])ret += t[i];
	}
	return ret;
}

/*subgraph DAG::border(const subgraph &s) const {
	subgraph ret;
	for(int i = 0 ; i < N ; i ++){
		if(!s[i])continue;
		for(int j: nex[i]){
			if(!s[j]){
				ret[i] = true;
				break;
			}
		}
	}
	return ret;
}*/

void in(DAG &dag,FILE* f){
	int hoge;
	dag.clear();
	int M;
	hoge = fscanf(f,"%d%d",&dag.N,&M);
	if(dag.N > MAX_N){
		puts("This graph is too large.");
		exit(1);
	}
	for(int i = 0 ; i < dag.N ; i ++){
		hoge = fscanf(f,"%d",&dag.t[i]);
	}
	for(int i = 0 ; i < dag.N ; i ++){
		hoge = fscanf(f,"%d",&dag.m[i]);
	}
	for(int i = 0 ; i < dag.N ; i ++){
		hoge = fscanf(f,"%d",&dag.p[i]);
	}
	for(int i = 0 ; i < M ; i ++){
		int a,b;
		hoge = fscanf(f,"%d%d",&a,&b);
		dag.add_edge(a,b);
	}
}

void FunctionNode::in(FILE* f){
	std::getline(std::cin,dir);
	fscanf(f,"%d%d",&k,&p);
	for(int i = 0 ; i < k ; i ++){
		int iid;
		std::string s;
		fscanf(f,"%d ",&iid);
		std::getline(std::cin,s);
		iID.push_back(iid);
		iType.push_back(s);
	}
	for(int i = 0 ; i < p ; i ++){
		int oid;
		fscanf(f,"%d\n",&oid);
		oID.push_back(oid);
	}
	std::getline(std::cin,FunctionType);
	std::getline(std::cin,Functiondetail);
}
FunctionNode nullfunc;

void topsor(const DAG &dag,std::vector<int> &ret){
	int V = dag.N;
	std::vector<int> used;
	for(int i = 0 ; i < V ; i ++){
		used.push_back(0);
	}
	for(int i = 0 ; i < V ; i ++){
		if(used[i])continue;
		std::stack<std::pair<int,int>> st;
		st.push(std::pair<int,int>(i,0));
		used[i] = 1;
		while(!st.empty()){
			std::pair<int,int> p = st.top(); st.pop();
			for(int j = p.sc ; j < dag.nex[p.fr].size() ; j ++){
				int w = dag.nex[p.fr][j];
				if(!used[w]){
					st.push(std::pair<int,int>(p.fr,j+1));
					st.push(std::pair<int,int>(w,0));
					used[w] = 1;
					goto next;
				}
			}
			ret.push_back(p.fr);
			next:;
		}
	}
	reverse(ret.begin(),ret.end());
}

void in_cg(DAG &dag, std::vector<FunctionNode> &Functions, FILE* f){
	int n,m;
	fscanf(f,"%d%d",&n,&m);
	dag.N = n;
	for(int i = 0 ; i < n ; i ++){
		int b;
		std::string s;
		fscanf(f,"%d%d",&b,&dag.m[i]);
		std::getline(std::cin,s);
		dag.p[i] = 0;
	}
	for(int i = 0 ; i < m ; i ++){
		FunctionNode func;
		func.id = i;
		func.in(f);
		Functions.push_back(func);
		if(func.dir == "forward"){
			if(func.p > 1){
				fprintf(stderr,"Function %d is forward but has multiple outputs.\n",i);
			}
			for(int a: func.iID){
				for(int b: func.oID){
					dag.add_edge(a,b);
				}
			}
			for(int b: func.oID){
				dag.t[b] = 1;
				if(func.FunctionType.size() >= 4 && func.FunctionType.substr(0,4) == "Conv"){
					dag.t[b] = 10;
				}
			}
		}
	}
	std::vector<int> vs;
	topsor(dag,vs);
	//topsor_check
	int id_topsor[MAX_N];
	for(int i = 0 ; i < vs.size() ; i ++){
		id_topsor[vs[i]] = i;
	}
	for(int i = 0 ; i < dag.N ; i ++){
		for(int to: dag.nex[i]){
			if(id_topsor[to] < id_topsor[i]){
				std::cerr << "topsor error" << std::endl;
				exit(1);
			}
		}
	}
	std::vector<FunctionNode> fo;
	std::vector<FunctionNode> b;
	nullfunc.id = -1;
	for(int i = 0 ; i < dag.N ; i ++){
		fo.push_back(nullfunc);
		b.push_back(nullfunc);
	}
	for(FunctionNode func: Functions){
		if(func.dir == "forward"){
			fo[func.oID[0]] = func;
		}
		else {
			b[func.id-Functions.size()/2] = func;
		}
	}
	std::vector<FunctionNode> f_;
	std::vector<FunctionNode> b_;
	for(int v: vs){
		if(fo[v].id != -1){
			f_.push_back(fo[v]);
			b_.push_back(b[fo[v].id]);
		}
	}
	//reverse(b_.begin(),b_.end());
	Functions = f_;
	for(FunctionNode func: b_){
		Functions.push_back(func);
	}
}

void show(const DAG &dag, std::string output){
	FILE* out = fopen( output.c_str() , "w" );
	fprintf(out,"digraph G{\n");
	fprintf(out,"clusterrank=local;\n");
	fprintf(out,"rankdir=LR;\n");
	fprintf(out,"subgraph cluster {\n");
	for(int i = 0 ; i < dag.N ; i ++){
		if(dag.p[i] >= 2)continue;
		fprintf(out,"%d;",i);
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
	fprintf(out,"subgraph cluster2 {\n");
	for(int i = 0 ; i < dag.N ; i ++){
		if(dag.p[i] <= 1)continue;
		fprintf(out,"%d;",i);
		for(int j: dag.nex[i]){
			if(dag.p[j] <= 1)continue;
			fprintf(out,"%d -> %d[dir=back];",j,i);
		}
		/*if(dag.nex_copy[i] != i){
			fprintf(out,"%d -> %d[style=dotted,arrowhead=none];",i,dag.nex_copy[i]);
			fprintf(out,"{rank = same; %d; %d;}",i,dag.nex_copy[i]);
		}*/
	}
	fprintf(out,"\n}\n");
	for(int i = 0 ; i < dag.N ; i ++){
		for(int j: dag.nex[i]){
			if((2*dag.p[i]-3)*(2*dag.p[j]-3) < 0){
				fprintf(out,"%d -> %d;",i,j);
			}
		}
	}
	fprintf(out,"\n}\n");
	fclose(out);
}


/*void show_subgraph(const DAG &dag, const subgraph& s, std::string output){
	FILE* out = fopen( output.c_str() , "w" );
	fprintf(out,"digraph G{\n");
	fprintf(out,"clusterrank=local;\n");
	fprintf(out,"rankdir=LR;\n");
	subgraph t = dag.border(s);
	for(int i = 0 ; i < dag.N ; i ++){
		if(t[i]){
			fprintf(out,"%d[style = filled, fillcolor = \"#FF0000\"]",i);
		}
		else if(s[i]){
			fprintf(out,"%d[style = filled, fillcolor = \"#FFA0A0\"]",i);
		}
	}
	fprintf(out,"subgraph cluster {\n");
	for(int i = 0 ; i < dag.N ; i ++){
		if(dag.p[dag.node_id[i]] >= 2)continue;
		fprintf(out,"%d;",i);
		for(int j: dag.nex_dep[i]){
			if(dag.p[dag.node_id[j]] >= 2)continue;
			fprintf(out,"%d -> %d;",i,j);
		}
		if(dag.nex_copy[i] != i){
			fprintf(out,"%d -> %d[style=dotted,arrowhead=none];",i,dag.nex_copy[i]);
			fprintf(out,"{rank = same; %d; %d;}",i,dag.nex_copy[i]);
		}
	}
	fprintf(out,"\n}\n");
	fprintf(out,"subgraph cluster2 {\n");
	for(int i = 0 ; i < dag.N ; i ++){
		if(dag.p[dag.node_id[i]] <= 1)continue;
		fprintf(out,"%d;",i);
		for(int j: dag.nex_dep[i]){
			if(dag.p[dag.node_id[j]] <= 1)continue;
			fprintf(out,"%d -> %d[dir=back];",j,i);
		}
		if(dag.nex_copy[i] != i){
			fprintf(out,"%d -> %d[style=dotted,arrowhead=none];",i,dag.nex_copy[i]);
			fprintf(out,"{rank = same; %d; %d;}",i,dag.nex_copy[i]);
		}
	}
	fprintf(out,"\n}\n");
	for(int i = 0 ; i < dag.N ; i ++){
		for(int j: dag.nex_dep[i]){
			if((2*dag.p[dag.node_id[i]]-3)*(2*dag.p[dag.node_id[j]]-3) < 0){
				fprintf(out,"%d -> %d;",i,j);
			}
		}
	}
	fprintf(out,"\n}\n");
	fclose(out);
}

void show_schedule(const DAG &dag, std::string output){
	//FILE* out = fopen( output.c_str() , "w" );
	for(int i = 0 ; i < dag.bestschedule.size() ; i ++){
		show_subgraph(dag,dag.bestschedule[i],output+std::to_string(i)+".dot");
	}
}
void show_schedule_on_initial_dag(const DAG &dag, const DAG &inidag, std::string output){
	subgraph s;
	subgraph t;
	for(int i = 0 ; i < dag.bestschedule.size() ; i ++){
		s.reset();
		t = dag.border(dag.bestschedule[i]);
		for(int j = 0 ; j < dag.N ; j ++){
			if(t[j]){
				s[dag.node_id[j]] = true;
			}
		}
		show_subgraph(inidag,s,output+std::to_string(i)+".dot");
	}
}*/