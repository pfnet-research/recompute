#ifndef DAGH
#define DAGH

#include<iostream>
#include<string>
#include<algorithm>
#include<set>
#include<vector>
#include<unordered_map>
#include<bitset>
#include<stack>
#include<queue>
#include<map>

const int MAX_N = 2500;
const int MAX_C = 100000000;
typedef std::bitset<MAX_N> subgraph;

#define fr first
#define sc second

class DAG{
public:
	int N;
	int m[MAX_N];
	int t[MAX_N];
	int p[MAX_N];
	std::vector<int> nex[MAX_N];
	std::vector<int> pre[MAX_N];
	
	void clear(int _N = MAX_N);
	void add_edge(int a,int b);
	
	std::vector<subgraph> cuts;
	std::unordered_map<subgraph,int> cut_id;
	std::vector<std::vector<int>> cut_nex;
	void make_cut_graph();
	
	subgraph delpl(const subgraph &s) const;
	subgraph delmi(const subgraph &s) const;
	int m_G(const subgraph &s) const;
	int t_G(const subgraph &s) const;
	
	//subgraph border(const subgraph &s) const;
	//subgraph redrob(const subgraph &s) const;
	//int m_G(const subgraph &s) const;
	//int t_G(const subgraph &s) const;
};
void in(DAG &dag, FILE *f = stdin);

void topsor(const DAG &dag, std::vector<int> &ret);

class FunctionNode{
public:
	int id;
	std::string dir; // forward or backward
	int k,p;
	std::vector<int> iID;
	std::vector<std::string> iType;
	std::vector<int> oID;
	std::string FunctionType, Functiondetail;
	void in(FILE*f = stdin);
};

void in_cg(DAG &dag, std::vector<FunctionNode> &Functions, FILE *f = stdin);

void show(const DAG &dag, std::string output = "graph.dot");
//void show_subgraph(const DAG &dag, const subgraph& s, std::string output = "subgraph.dot");
//void show_schedule(const DAG &dag, std::string output = "schedule");
//void show_schedule_on_initial_dag(const DAG &dag, const DAG &inidag, std::string output = "schedule_on_dag");

#endif
