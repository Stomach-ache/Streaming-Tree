/*
  Code written by : Sujay Khandagale and Han Xiao

  The code is based on the codebase written by Yashoteja Prabhu for Parabel published at WWW'18.
*/

/* The header contains only one modification to the tree structure
that is an internal node can have mutiple children
(instead of only two in parabel) */

#ifndef BONSAI_H
#define BONSAI_H

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <functional>
#include <unordered_set>
#include <unordered_map>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cassert>



#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "svm.h"

#define DBG if(false)

#include "helpers.h"

using namespace std;

enum _Septype { L2R_L2LOSS_SVC=0, L2R_LR };
enum _Parttype { KMEANS=0, EXTERNAL };

struct subproblem
{
	int l, n;
	int *y;
	SMatF *x;
	double bias;            /* < 0 if no bias term */
};

class Node
{
 public:
  _bool is_leaf;
  vector<_int> children;
  _int pos_child;  // dummy
  _int neg_child;
  _int depth;
  VecI Y;
  VecF Y_cent;
  SMatF* w;
  //float *alpha
  VecIF X; // store the instance ids to predict (for test data), (id, score)

  Node()
    {
      is_leaf = false;
      // pos_child = neg_child = -1;
      depth = 0;
      w = NULL;
    }

  Node( VecI Y, _int depth, _int max_depth )
    {
      this->Y = Y;
      this->depth = depth;
      // this->pos_child = -1;
      // this->neg_child = -1;
      this->is_leaf = (depth >= max_depth-1);
      this->w = NULL;
    }

  ~Node()
    {
      delete w;
    }

  _float get_ram()
  {
    _float ram = 0;
    ram += sizeof( Node );
    ram += sizeof( _int ) * Y.capacity();
    ram += sizeof(_float) * Y_cent.capacity();
    ram += w->get_ram();
    return ram;
  }

  friend ostream& operator<<( ostream& fout, const Node& node )
  {
    fout << node.is_leaf << "\n";
    for(_int i=0; i < node.children.size(); i++){
      _int c = node.children[i];
      if (i != (node.children.size()-1))
        fout << c << " ";
      else
        fout << c << "\n";
    }
    // fout << node.pos_child << " " << node.neg_child << "\n";
    fout << node.depth << "\n";

    fout << node.Y.size();
    for( _int i=0; i<node.Y.size(); i++ )
      fout << " " << node.Y[i];
    fout << "\n";

    fout << node.Y_cent.size();
    for( _int i=0; i<node.Y_cent.size(); i++ )
      fout << " " << node.Y_cent[i];
    fout << "\n";

    assert (node.w != nullptr);
    fout << (*node.w);
  }

  friend istream& operator>>( istream& fin, Node& node )
  {
    fin >> node.is_leaf;
    DBG {      cout << "node>>: is_leaf = " << node.is_leaf << endl;	}

    if(!node.is_leaf){
      string line;

      getline(fin, line);
      getline(fin, line);
      DBG {cout << "node>>: line=" << line << endl;}
      /*
	0
	1 2 3 4 5 6 7 8 9 10
	0
      */
      istringstream is(line);
      _int c = -1;
      DBG {cout << "node>>: child = " << endl;}
      while(is >> c) {
	node.children.push_back(c);
	DBG {cout << c << ", ";}
      }

      DBG {cout << endl;}
    }

    // fin >> node.pos_child >> node.neg_child;
    fin >> node.depth;
    DBG {cout << "node>> node.depth= " << node.depth << endl;}

    _int Y_size;
    fin >> Y_size;
    node.Y.resize( Y_size );

    for( _int i=0; i<Y_size; i++ )
      fin >> node.Y[i];

    _int Y_cent_size;
    fin >> Y_cent_size;
    node.Y_cent.resize( Y_cent_size );

    for( _int i=0; i<Y_cent_size; i++ )
      fin >> node.Y_cent[i];

    node.w = new SMatF;
    fin >> (*node.w);
  }
};

class Tree
{
 public:
  _int num_Xf;
  _int num_Y;
  vector<Node*> nodes;

  Tree()
    {

    }

  Tree( string model_dir, _int tree_no )
    {
      DBG {cout << "read tree started" << endl;}
      ifstream fin;
      fin.open( model_dir + "/" + to_string( tree_no ) + ".tree" );
      DBG {cout << "read tree done" << endl;}

      fin >> num_Xf;
      fin >> num_Y;
      _int num_node;
      fin >> num_node;

      for( _int i=0; i<num_node; i++ )
	{
	  Node* node = new Node;
	  nodes.push_back( node );
	}

      for( _int i=0; i<num_node; i++ ) {
	fin >> (*nodes[i]);
	DBG {cout << "read node " << i << " done" << endl;}
      }

      fin.close();
    }

  ~Tree() {
    for(_int i=0; i<nodes.size(); i++)
      delete nodes[i];
  }

  _float get_ram()
  {
    _float ram = 0;
    ram += sizeof( Tree );
    for(_int i=0; i<nodes.size(); i++)
      ram += nodes[i]->get_ram();
    return ram;
  }

  void write( string model_dir, _int tree_no )
  {
    ofstream fout;
    fout.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

    fout << num_Xf << "\n";
    fout << num_Y << "\n";
    _int num_node = nodes.size();
    fout << num_node << "\n";

    for( _int i=0; i<num_node; i++ ) {
      fout << (*nodes[i]);
    }

    fout.close();
  }
};

class Param
{
 public:
  _int num_trn;
  _int num_Xf;
  _int num_Y;
  _float log_loss_coeff;
  _int max_leaf;  //  dummy
  _int max_depth;
  _float bias;
  _int start_tree;
  _int num_tree;
  _int num_children;
  _float svm_th;
  _float cent_th;
  _float kmeans_eps;
  _int num_thread;
  _int beam_size;
  _float discount;
  _int svm_iter;
  _bool quiet;
  _Septype septype;
  _int cent_type;

  _Parttype part_type;
  string label_graph_path;
  _int metis_ufactor;

  Param()
    {
      num_Xf = 0;
      num_Y = 0;
      log_loss_coeff = 1.0;
      max_depth = 2;
      bias = 1.0;
      start_tree = 0;
      num_tree = 3;
      num_children = 10;
      svm_th = 0.1;
      cent_th = 0;
      kmeans_eps = 1e-4;
      num_thread = 1;
      beam_size = 10;
      discount = 0.98;
      svm_iter = 20;
      quiet = false;
      septype = L2R_L2LOSS_SVC;
      part_type = KMEANS;
      label_graph_path = "";
      metis_ufactor = 30;
      cent_type=0;
    }

  Param(string fname)
    {
      check_valid_filename(fname,true);
      ifstream fin;
      fin.open(fname);

      fin>>num_Xf;
      fin>>num_Y;
      fin>>log_loss_coeff;
      fin>>max_depth;
      fin>>bias;
      fin>>start_tree;
      fin>>num_tree;
      fin>>num_children;
      fin>>svm_th;
      fin>>cent_th;
      fin>>kmeans_eps;
      fin>>num_thread;
      fin>>beam_size;
      fin>>discount;
      fin>>svm_iter;
      fin>>quiet;
      _int st;
      fin>>st;
      septype = (_Septype)st;

      _int pt;
      fin>>pt;
      part_type = (_Parttype)pt;

      fin>>label_graph_path;

      fin>>metis_ufactor;

      fin.close();
    }

  void write(string fname)
  {
    check_valid_filename(fname,false);
    ofstream fout;
    fout.open(fname);

    fout<<num_Xf<<"\n";
    fout<<num_Y<<"\n";
    fout<<log_loss_coeff<<"\n";
    fout<<max_depth<<"\n";
    fout<<bias<<"\n";
    fout<<start_tree<<"\n";
    fout<<num_tree<<"\n";
    fout<<num_children<<"\n";
    fout<<svm_th<<"\n";
    fout<<cent_th<<"\n";
    fout<<kmeans_eps<<"\n";
    fout<<num_thread<<"\n";
    fout<<beam_size<<"\n";
    fout<<discount<<"\n";
    fout<<svm_iter<<"\n";
    fout<<quiet<<"\n";
    fout<<septype<<"\n";
    fout<<part_type<<"\n";
    fout<<label_graph_path<<"\n";
    fout<<metis_ufactor<<"\n";
    fout.close();
  }
};

void append_bias( SMatF* mat, _float bias );
_float mult_s_s_vec( pairIF* v1, pairIF* v2, _int size1, _int size2 );
_float mult_d_s_vec( _float* dvec, pairIF* svec, _int siz );
void add_s_to_d_vec( pairIF* svec, _int siz, _float* dvec );
void div_d_vec_by_scalar( _float* dvec, _int siz, _float s );
void normalize_d_vec( _float* dvec, _int siz );
void split_node_kmeans( Node* node, SMatF* X_Xf, SMatF* Y_X, SMatF* cent_mat, _int nr, VecI& n_Xf, VecI& partition, Param& param, int tree_no );
void train_leaf_svms( Node* node, SMatF* X_Xf, SMatF* Y_X, _int nr, VecI& n_Xf, Param& param );
void split_node_kmeans( Node* node, SMatF* X_Xf, SMatF* Y_X, SMatF* cent_mat, _int nr, VecI& n_Xf, VecI& partition, Param& param );
void shrink_data_matrices_with_cent_mat( SMatF* trn_X_Xf, SMatF* trn_Y_X, SMatF* cent_mat, VecI& n_Y, Param& param, SMatF*& n_trn_X_Xf, SMatF*& n_trn_Y_X, SMatF*& n_cent_mat, VecI& n_X, VecI& n_Xf, VecI& n_cXf );
void shrink_data_matrices( SMatF* trn_X_Xf, SMatF* trn_Y_X, VecI& n_Y, Param& param, SMatF*& n_trn_X_Xf, SMatF*& n_trn_Y_X, VecI& n_X, VecI& n_Xf);
void squeeze_partition_array(vector<_int> &partition);
void solve_L2R_L2LOSS_SVC( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter, bool reset_w );
void solve_l2r_l1l2_svc( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter, bool reset_w );
void solve_l2r_lr_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter, bool reset_w );
void reindex_rows( SMatF* mat, _int nr, VecI& rows, int base_no );
SMatF* partition_to_assign_mat( SMatF* Y_X, VecI& partition, int base_no );
SMatF* finetune_svms( SMatF *prev_w_mat, SMatF* trn_X_Xf, SMatF* trn_Y_X, Param& param, int finetune_iter, int nr, VecI &n_Xf );
SMatF* svms( SMatF* trn_X_Xf, SMatF* trn_Y_X, Param& param, int base_no );

Tree* train_tree( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _int tree_no );
void train_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, SMatF* trn_X_XY, Param& param, string model_dir, _float& train_time );

void update_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, SMatF* trn_X_XY, Param& param, string model_dir, _float& train_time, int base_no );

SMatF* predict_tree( SMatF* tst_X_Xf, Tree* tree, Param& parami, int lft, int rgt );
SMatF* predict_trees( SMatF* tst_X_Xf, Param& param, string model_dir, _float& prediction_time, _float& model_size, int lft, int rgt );


extern mutex mtx;
extern thread_local mt19937 reng; // random number generator used during training
extern thread_local _bool* mask; // shared among threads?

#endif
