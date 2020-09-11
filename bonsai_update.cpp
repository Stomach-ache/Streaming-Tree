#include <bonsai.h>
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

    stable_sort(begin(nodes), end(nodes), [&] (Node *a, Node *b) {
            return a->depth < b->depth;
            });

    vector<int> reverse_nodes_id(num_node);
    for (int i = 0; i < num_node; ++ i) {
        reverse_nodes_id[nodes_id[i]] = i;
    }

    for (int i = 0; i < num_node; ++ i) {
        for (int j = 0; j < nodes[i].children.size(); ++ j) {
            nodes[i]->children[j] = reverse_nodes_id[nodes[i]->children[j]];
        }
    }
}

void update_tree(SMatF *X_Xf, SMatF *Y_X,SMatF *cent_mat, Tree *tree, Param &param, int tree_no) {
    // base_no: the number of labels already observed
    // cent_mat: label representation

    int num_Y = Y_X->nc;
    for (int i = 0; i < num_Y; ++ i) {

        // root
        int cur_node = 0;

        while  (true) {
            nodes[cur_node]->n_Y.push_back(i + base_no);
            float *node_cent = new float[cent_mat->nr];
            for (int ch: nodes[cur_node]->n_Y) {
                add_s_to_d_vec(cent_mat->data[ch], cent_mat->size[ch], node_cent);
            }
            normalize_d_vec(node_cent, cent_mat->nr);

            if (nodes[cur_node]->is_leaf == false) {

                int maxCh = 0;
                float maxSim = -1;
                for (int ch: nodes[cur_node]->children) {
                    float cos_sim = mult_d_s_vec(node_cent, cent_mat[i + base_no]);
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

    // check if leaf nodes need further split
    //
    int num_nodes = nodes.size();
    for (int i = 0; i < num_nodes; ++ i) {
        if (nodes[i]->is_leaf == true) {
            if (nodes[i]->n_Y.size() > param.num_children) {
                nodes[i]->is_leaf = false;

                // split node
                split_node_kmeans( node, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, num_Xf, n_Xf, partition, param );

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
                    if(child_labels.size() <= param.num_children)
                        child_node->is_leaf = true;

                    nodes.push_back( child_node );
                    node->children.push_back( nodes.size()-1 );
                }

            } else {
                // update leaf classifier
                train_leaf_svms(nodes[i], X_Xf, Y_X, nr, n_Xf, param);
            }
        }
    }


    // rearrange nodes
    // sort_nodes();
}

void update_trees_thread( SMatF* tst_X_Xf, SMatF *trn_Y_X, SMatF *cent_mat, Param param, _int s, _int t, string &model_dir) {
    Timer timer;

    for(_int i=s; i<s+t; i++) {
        timer.resume();
        cout<<"tree "<<i<<" training started"<<endl;

        Tree* tree = new Tree( model_dir, i );
        update_tree(tree, trn_X_Xf, trn_Y_X, cent_mat, param, i );
        timer.stop();

        tree->write( model_dir, i );

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

void update_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, SMatF* trn_X_XY, Param& param, string model_dir, _float& train_time ) {
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

    append_bias( trn_X_Xf, param.bias );

    _int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
    vector<thread> threads;
    _int s = param.start_tree; // the tree id?
    for( _int i=0; i<param.num_thread; i++ )
    {
        if( s < param.start_tree+param.num_tree )
        {
            _int t = min( tree_per_thread, param.start_tree+param.num_tree-s );
            threads.push_back( thread( update_trees_thread, trn_X_Xf, trn_Y_X, cent_mat, param, s, t, model_dir, ref(t_time) ));
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
