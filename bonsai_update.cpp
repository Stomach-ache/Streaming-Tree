#include "bonsai.h"
#include <vector>
#include <algorithm>

using namespace std;

void sort_nodes(vector<Node*> &nodes) {
    int num_node = nodes.size();

    vector<int> nodes_id(num_node);
    for (int i = 0; i < num_node; ++ i) nodes_id[i] = i;

    stable_sort(begin(nodes_id), end(nodes_id), [&nodes] (int i, int j) {
            return nodes[i]->depth < nodes[j]->depth;
            });

    stable_sort(begin(nodes), end(nodes), [] (Node *a, Node *b) {
            return a->depth < b->depth;
            });

    vector<int> reverse_nodes_id(num_node);
    for (int i = 0; i < num_node; ++ i) {
        reverse_nodes_id[nodes_id[i]] = i;
    }

//    for (int i = 0; i < 10; ++ i) cout << i << ' ' << reverse_nodes_id[i] << endl;

    cout << "build reverse nodes id done..." << endl;

    for (int i = 0; i < num_node; ++ i) {
        for (int j = 0; j < nodes[i]->children.size(); ++ j) {
            nodes[i]->children[j] = reverse_nodes_id[nodes[i]->children[j]];
        }
    }

}

void update_tree(SMatF *trn_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, Tree *tree, Param &param, int tree_no, int base_no) {
    // base_no: the number of labels already observed
    // cent_mat: label representation
    //
    reng.seed(tree_no);
    _int num_X = trn_X_Xf->nc;
    _int num_Xf = trn_X_Xf->nr;
    _int num_Y = trn_Y_X->nc;
    _int num_XY = cent_mat->nr;

    // update tree attribute
    tree->num_Y = num_Y;

    vector<Node*> &nodes = tree->nodes;
    _int max_n = max( max( max( num_X+1, num_Xf+1 ), num_Y+1 ), num_XY+1);
    mask = new _bool[ max_n ]();
    float *node_cent = new float[cent_mat->nr];

    cout << "number of labels = " << num_Y << ", base_no = " << base_no << endl;
    for (int i = base_no; i < num_Y; ++ i) {

        // root
        int cur_node = 0;

        while  (true) {
            nodes[cur_node]->Y.push_back(i);
            if (nodes[cur_node]->is_leaf == false) {

                int maxCh = 0;
                float maxSim = -1;

                for (int ch: nodes[cur_node]->children) {

                    for (int j = 0; j < cent_mat->nr; ++ j) node_cent[j] = 0;

                    for (int lbl: nodes[ch]->Y) {
                        add_s_to_d_vec(cent_mat->data[lbl], cent_mat->size[lbl], node_cent);
                    }
                    normalize_d_vec(node_cent, cent_mat->nr);

                    float cos_sim = mult_d_s_vec(node_cent, cent_mat->data[i], cent_mat->size[i]);
                    //cout << "child = " << ch << " cos_sim = " << cos_sim << endl;
                    if (cos_sim > maxSim) {
                        maxSim = cos_sim;
                        maxCh = ch;
                    }

                }

                cur_node = maxCh;
            } else {
                break;
            }
        }
        //
    }

    cout << "label insertion done..." << endl;

    // check if leaf nodes need further split
    //
    int num_nodes = nodes.size();
    int tmp_num_nodes = num_nodes;
    _int max_depth = param.max_depth;
    for (int i = 0; i < tmp_num_nodes; ++ i) {
        //cout << "visit node: " << i << endl;

        Node *node = nodes[i];
        //cout << "visit leaf node: " << i << endl;
        VecI& n_Y = node->Y; // labels in node, vector of integer
        SMatF* n_trn_X_Xf = NULL; // feature matrix in node
        SMatF* n_trn_Y_X = NULL; // label matrix in node
        SMatF* n_cent_mat = NULL; // centroid matrix in node
        VecI n_X;
        VecI n_Xf;
        VecI n_cXf;

        // slice the matrix by rows and columns
        shrink_data_matrices_with_cent_mat( trn_X_Xf, trn_Y_X, cent_mat, n_Y, param, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, n_X, n_Xf, n_cXf );
        //cout << "slicing matrix done..." << endl;

        if (node->is_leaf == true) {
            if (node->Y.size() > param.num_children && i < num_nodes && node->depth + 1 < max_depth) {
                //cout << "ready to partition leaf..." << endl;
                node->is_leaf = false;

                // split node
                VecI partition; // partitioning starting from 0
                split_node_kmeans( node, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, num_Xf, n_Xf, partition, param, tree_no );

                int n_effective_partitions = unordered_set<_int>(partition.begin(), partition.end()).size();

                cout << "n_effective_partitions=" << n_effective_partitions << endl;

                vector< vector<_int> > labels_by_child(n_effective_partitions);
                for( _int j=0; j<n_Y.size(); j++){
                    assert(partition[j] >= 0);
                    assert(partition[j] < n_effective_partitions);
                    // cout << "partition[j]=" << partition[j] << endl;
                    // cout << "param.num_children=" << param.num_children << endl;
                    labels_by_child[ partition[j] ].push_back( n_Y[j] );
                }

                for(vector<_int>  child_labels: labels_by_child) {
                    Node* child_node = new Node( child_labels, node->depth+1, max_depth );

                    // when not enough labels to partition, make it a leaf
                    //if(child_labels.size() <= param.num_children)
                    child_node->is_leaf = true;

                    nodes.push_back( child_node );
                    node->children.push_back( nodes.size()-1 );
                    ++ tmp_num_nodes;
                }

            } else {
                //cout << "ready to update leaf..." << endl;
                //update leaf classifier
                //
                train_leaf_svms( node, n_trn_X_Xf, n_trn_Y_X, num_Xf, n_Xf, param );
                //assert (node->w != nullptr);
            }
        } else {
            /*
            VecI partition(node->Y.size());
            for (int j = 0; j < node->children.size(); ++ j) {
                int ch = node->children[j];
                for (int l: nodes[ch]->Y) partition[l] = j;
            }
            SMatF* assign_mat = partition_to_assign_mat( n_trn_Y_X, partition );
            //node->w = finetune_svms( node->w, n_trn_X_Xf, assign_mat, param, 1, num_Xf, n_Xf );
            delete node->w;
            node->w = svms(n_trn_X_Xf, assign_mat, param, 0);
            delete assign_mat;
            */
        }
        // classifier normalization
        node->w->unit_normalize_columns();
    }

    cout << "leaf partition done..." << endl;
    // rearrange nodes
    sort_nodes(nodes);
    delete node_cent;
    delete[] mask;
    cout << "nodes sorting done..." << endl;
}

void update_trees_thread( SMatF* trn_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, Param param, _int s, _int t, string model_dir, _float *train_time, int base_no) {
    Timer timer;

    for(_int i=s; i<s+t; i++) {
        timer.resume();
        cout<<"tree "<<i<<" training started"<<endl;

        Tree* tree = new Tree( model_dir, i );
        update_tree(trn_X_Xf, trn_Y_X, cent_mat, tree, param, i, base_no);
        timer.stop();

        cout << "tree write starts..." << endl;
        tree->write( model_dir, i );
        cout << "tree write done..." << endl;

        timer.resume();
        delete tree;

        cout<<"tree "<<i<<" training completed"<<endl;
        timer.stop();
    }
    {
        timer.resume();
        lock_guard<mutex> lock(mtx);
        *train_time += timer.stop();
    }
}

void update_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, SMatF* trn_X_XY, Param& param, string model_dir, _float& train_time, int base_no = 0 ) {
    // called by main
    // train trees in parallel
    _float* t_time = new _float;
    *t_time = 0;
    Timer timer;

    timer.start();
    param.num_trn = trn_X_Xf->nc;
    trn_X_Xf->unit_normalize_columns();
    SMatF* trn_Y_X = trn_X_Y->transpose(); // each column a training sample

    SMatF* cent_mat = NULL;

        // cent_mat = trn_X_Xf->prod( trn_Y_X ); // get the label matrix , each column a label
        // cent_mat->unit_normalize_columns();
        // cent_mat->threshold( param.cent_th ); // make it sparse by thresholding

    if(param.cent_type == 0)
    {
        cent_mat = trn_X_Xf->prod( trn_Y_X );
        cent_mat->unit_normalize_columns();
    }

    else if(param.cent_type == 1)
    {
        cent_mat = trn_X_Y->prod( trn_Y_X );
        cent_mat->remove_self_coocc(0);  //passing 0 instead of param.num_Xf
        cent_mat->unit_normalize_columns();
    }

    else if(param.cent_type == 2)
    {
        cent_mat = trn_X_XY->prod( trn_Y_X ); // get the label matrix , each column a label
        // cent_mat->unit_normalize_columns();
        cent_mat->unit_normalize_X_columns(param.num_Xf, param.num_Y);  //changed
        // cent_mat->unit_normalize_Y_columns(param.num_Xf, param.num_Y);
        // cent_mat->normalize_Y_columns(param.num_Xf, param.num_Y);
        cent_mat->remove_self_coocc(param.num_Xf);
        cent_mat->unit_normalize_Y_columns(param.num_Xf, param.num_Y);
        // cent_mat->make_coooc_cons(param.num_Xf, 1);
    }

    cent_mat->threshold( param.cent_th ); // make it sparse by thresholding

    //append_bias( trn_X_Xf, param.bias );

    _int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
    vector<thread> threads;
    _int s = param.start_tree; // the tree id?
    for( _int i=0; i<param.num_thread; i++ )
    {
        if( s < param.start_tree+param.num_tree )
        {
            _int t = min( tree_per_thread, param.start_tree+param.num_tree-s );
            threads.push_back( thread( update_trees_thread, trn_X_Xf, trn_Y_X, cent_mat, param, s, t, model_dir, ref(t_time), base_no ));
            s += t;
        }
    }
    timer.stop();

    for(_int i=0; i<threads.size(); i++)
        threads[i].join();

    timer.resume();
    delete trn_Y_X;
    delete cent_mat;

    *t_time += timer.stop();
    train_time = *t_time;
    delete t_time;
}
