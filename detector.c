#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#include "capture.h"
#include <malloc.h>
#include <windows.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "dark_cuda.h"
#include "blas.h"
#include "connected_layer.h"
#include "image.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"
int check_mistakes;

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *valid_images = option_find_str(options, "valid", train_images);
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    network net_map;
    if (calc_map) {
        FILE* valid_file = fopen(valid_images, "r");
        if (!valid_file) {
            printf("\n Error: There is no %s file for mAP calculation!\n Don't use -map flag.\n Or set valid=%s in your %s file. \n", valid_images, train_images, datacfg);
            getchar();
            exit(-1);
        }
        else fclose(valid_file);

        cuda_set_device(gpus[0]);
        printf(" Prepare additional network for mAP calculation...\n");
        net_map = parse_network_cfg_custom(cfgfile, 1, 1);

        int k;  // free memory unnecessary arrays
        for (k = 0; k < net_map.n; ++k) {
            free_layer(net_map.layers[k]);
        }
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network* nets = (network*)calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for (i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if (weightfile) {
            load_weights(&nets[i], weightfile);
        }
        if (clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }
    else if (actual_batch_size < 64) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    int init_w = net.w;
    int init_h = net.h;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_batch(net);
    iter_save_last = get_current_batch(net);
    iter_map = get_current_batch(net);
    float mean_average_precision = -1;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

#ifdef OPENCV
    args.threads = 3 * ngpus;   // Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    //args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = 5;
    int number_of_lines = 100;
    int img_size = 1000;
    img = draw_train_chart(max_img_loss, net.max_batches, number_of_lines, img_size, dont_show);
#endif    //OPENCV
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        args.threads = net.subdivisions * ngpus;    // 2 * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    //printf(" imgs = %d \n", imgs);

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while (get_current_batch(net) < net.max_batches) {
        if (l.random && count++ % 10 == 0) {
            printf("Resizing\n");
            float random_val = rand_scale(1.4);    // *x or /x
            int dim_w = roundl(random_val*init_w / 32 + 1) * 32;
            int dim_h = roundl(random_val*init_h / 32 + 1) * 32;

            // at the beginning
            if (avg_loss < 0) {
                dim_w = roundl(1.4*init_w / 32 + 1) * 32;
                dim_h = roundl(1.4*init_h / 32 + 1) * 32;
            }

            if (dim_w < 32) dim_w = 32;
            if (dim_h < 32) dim_h = 32;

            printf("%d x %d \n", dim_w, dim_h);
            args.w = dim_w;
            args.h = dim_h;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (i = 0; i < ngpus; ++i) {
                resize_network(nets + i, dim_w, dim_h);
            }
            net = nets[0];
        }
        time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        if(!b.x) break;
        printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
        }
        image im = float_to_image(448, 448, 3, train.X.vals[10]);
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
        draw_bbox(im, b, 8, 1,0,0);
        }
        save_image(im, "truth11");
        */

        printf("Loaded: %lf seconds\n", (what_time_is_it_now() - time));

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);

        int calc_map_for_each = 4 * train_images_num / (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
        calc_map_for_each = fmax(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = fmax(next_map_calc, net.burn_in);
        next_map_calc = fmax(next_map_calc, 1000);
        if (calc_map) {
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@0.5 = %2.2f %% ", mean_average_precision * 100);
        }

        if (net.cudnn_half) {
            if (i < net.burn_in * 3) fprintf(stderr, "\n Tensor Cores are disabled until the first %d iterations are reached.", 3 * net.burn_in);
            else fprintf(stderr, "\n Tensor Cores are used.");
        }
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), i*imgs);

        int draw_precision = 0;
        if (calc_map && (i >= next_map_calc || i == net.max_batches)) {
            if (l.random) {
                printf("Resizing to initial size: %d x %d \n", init_w, init_h);
                args.w = init_w;
                args.h = init_h;
                pthread_join(load_thread, 0);
                free_data(train);
                train = buffer;
                load_thread = load_data(args);
                int k;
                for (k = 0; k < ngpus; ++k) {
                    resize_network(nets + k, init_w, init_h);
                }
                net = nets[0];
            }

            copy_weights_net(net, &net_map);

            // combine Training and Validation networks
            //network net_combined = combine_train_valid_networks(net, net_map);

            iter_map = i;
            mean_average_precision = validate_detector_map(datacfg, cfgfile, weightfile, 0.25, 0.5, 0, &net_map);// &net_combined);
            printf("\n mean_average_precision (mAP@0.5) = %f \n", mean_average_precision);
            draw_precision = 1;
        }
#ifdef OPENCV
        draw_train_loss(img, img_size, avg_loss, max_img_loss, i, net.max_batches, mean_average_precision, draw_precision, "mAP%", dont_show, mjpeg_port);
#endif    // OPENCV

        //if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
        //if (i % 100 == 0) {
        if (i >= (iter_save + 1000) || i % 1000 == 0) {
            iter_save = i;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }

        if (i >= (iter_save_last + 100) || i % 100 == 0) {
            iter_save_last = i;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (i = 0; i < ngpus; ++i) free_network(nets[i]);
    free(nets);
    //free_network(net);

    if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c) p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for (i = 0; i < num_boxes; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int myclass = j;
            if (dets[i].prob[myclass]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass],
                xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    }
    else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    }
    else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE**)calloc(classes, sizeof(FILE*));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i = 0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)calloc(nthreads, sizeof(image));
    image* val_resized = (image*)calloc(nthreads, sizeof(image));
    image* buf = (image*)calloc(nthreads, sizeof(image));
    image* buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;
    //args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            char *path = paths[i + t - nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            int letterbox = (args.type == LETTERBOX_DATA);
            detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letterbox);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco) {
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            }
            else if (imagenet) {
                print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
            }
            else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for (j = 0; j < classes; ++j) {
        if (fps) fclose(fps[j]);
    }
    if (coco) {
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)time(0) - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    srand(time(0));

    //list *plist = get_paths("data/coco_val_5k.list");
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    //layer l = net.layers[net.n - 1];

    int j, k;

    int m = plist->size;
    int i = 0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for (i = 0; i < m; ++i) {
        char *path = paths[i];
        image orig = load_image(path, 0, 0, net.c);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        int letterbox = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (k = 0; k < nboxes; ++k) {
            if (dets[k].objectness > thresh) {
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
            float best_iou = 0;
            for (k = 0; k < nboxes; ++k) {
                float iou = box_iou(dets[k].bbox, t);
                if (dets[k].objectness > thresh && iou > best_iou) {
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if (best_iou > iou_thresh) {
                ++correct;
            }
        }
        //fprintf(stderr, " %s - %s - ", paths[i], labelpath);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(box_prob *)pa;
    box_prob b = *(box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, network *existing_net)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network net;
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    char **paths_dif = NULL;
    if (difficult_valid_images) {
        list *plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }


    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .005;
    const float nms = .45;
    //const float iou_thresh = 0.5;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)calloc(nthreads, sizeof(image));
    image* val_resized = (image*)calloc(nthreads, sizeof(image));
    image* buf = (image*)calloc(nthreads, sizeof(image));
    image* buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;
    //args.type = LETTERBOX_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob* detections = (box_prob*)calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int* truth_classes_count = (int*)calloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = (float*)calloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = (int*)calloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = (int*)calloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);

            int nboxes = 0;
            float hier_thresh = 0;
            detection *dets;
            if (args.type == LETTERBOX_DATA) {
                int letterbox = 1;
                dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
            }
            else {
                int letterbox = 0;
                dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letterbox);
            }
            //detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letterbox); // for letterbox=1
            if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

            char labelpath[4096];
            replace_image_to_label(path, labelpath);
            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            int i, j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;
            }

            // difficult
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                replace_image_to_label(path_dif, labelpath_dif);

                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }

            const int checkpoint_detections_count = detections_count;

            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = (box_prob*)realloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                            //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                        else {
                            // if object is difficult then remove detection
                            for (j = 0; j < num_labels_dif; ++j) {
                                box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
                                float current_iou = box_iou(dets[i].bbox, t);
                                if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
                                    --detections_count;
                                    break;
                                }
                            }
                        }

                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }
                            }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                                avg_iou_per_class[class_id] += max_iou;
                                tp_for_thresh_per_class[class_id]++;
                            }
                            else{
                                fp_for_thresh++;
                                fp_for_thresh_per_class[class_id]++;
                            }
                        }
                    }
                }
            }

            unique_truth_count += num_labels;

            //static int previous_errors = 0;
            //int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
            //int errors_in_this_image = total_errors - previous_errors;
            //previous_errors = total_errors;
            //if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
            //char buff[1000];
            //sprintf(buff, "%s\n", path);
            //if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }

    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for(class_id = 0; class_id < classes; class_id++){
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)calloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)calloc(detections_count, sizeof(pr_t));
    }
    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int* detection_per_class_count = (int*)calloc(classes, sizeof(int));
    for (j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)calloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010?012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0)
        {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank)
            {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
        }
        // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else
        {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points-1);
                double cur_precision = 0;
                for (rank = 0; rank < detections_count; ++rank)
                {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        printf("class_id = %d, name = %s, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
            i, names[i], avg_precision * 100, tp_for_thresh_per_class[i], fp_for_thresh_per_class[i]);

        float class_precision = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
        float class_recall = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);

        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision, mean_average_precision * 100);

    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);

    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
    printf("\nSet -points flag:\n");
    printf(" `-points 101` for MS COCO \n");
    printf(" `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) \n");
    printf(" `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset\n");
    if (reinforcement_fd != NULL) fclose(reinforcement_fd);

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    if (existing_net) {
        //set_batch_network(&net, initial_batch);
    }
    else {
        free_network(net);
    }

    return mean_average_precision;
}

typedef struct {
    float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(anchors_t *)pa;
    anchors_t b = *(anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int anchors_data_comparator(const float **pa, const float **pb)
{
    float *a = (float *)*pa;
    float *b = (float *)*pb;
    float diff = b[0] * b[1] - a[0] * a[1];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}


void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }

    //float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
    float* rel_width_height_array = (float*)calloc(1000, sizeof(float));


    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    srand(time(0));
    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);
        printf(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        //printf(" new path: %s \n", labelpath);
        char buff[1024];
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
                if (check_mistakes) getchar();
            }
            number_of_boxes++;
            rel_width_height_array = (float*)realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
        }
    }
    printf("\n all loaded. \n");
    printf("\n calculating k-means++ ...");

    matrix boxes_data;
    model anchors_data;
    boxes_data = make_matrix(number_of_boxes, 2);

    printf("\n");
    for (i = 0; i < number_of_boxes; ++i) {
        boxes_data.vals[i][0] = rel_width_height_array[i * 2];
        boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
        //if (w > 410 || h > 410) printf("i:%d,  w = %f, h = %f \n", i, w, h);
    }

    // Is used: distance(box, centroid) = 1 - IoU(box, centroid)

    // K-means
    anchors_data = do_kmeans(boxes_data, num_of_clusters);

    qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

    //gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
    //float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
        float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
                                                         //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        float best_iou = 0;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
            float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
            float min_w = (box_w < anchor_w) ? box_w : anchor_w;
            float min_h = (box_h < anchor_h) ? box_h : anchor_h;
            float box_intersect = min_w*min_h;
            float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
            float iou = box_intersect / box_union;
            float distance = 1 - iou;
            if (distance < min_dist) min_dist = distance, cluster_idx = j, best_iou = iou;
        }

        float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
        float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
        if (best_iou > 1 || best_iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %f, box_h = %f, anchor_w = %f, anchor_h = %f, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, best_iou);
        }
        else avg_iou += best_iou;
    }
    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);

    char buff[1024];
    FILE* fw = fopen("anchors.txt", "wb");
    if (fw) {
        printf("\nSaving anchors to the file: anchors.txt \n");
        printf("anchors = ");
        for (i = 0; i < num_of_clusters; ++i) {
            float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
            float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
            if (width > 32) sprintf(buff, "%3.0f,%3.0f", anchor_w, anchor_h);
            else sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fw);
            if (i + 1 < num_of_clusters) {
                fwrite(", ", sizeof(char), 2, fw);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fw);
    }
    else {
        printf(" Error: file anchors.txt can't be open \n");
    }

    if (show) {
#ifdef OPENCV
        //show_acnhors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
#endif // OPENCV
    }
    free(rel_width_height_array);

    getchar();
}



/*************************************************************************************************************/
//#include <C:\MinGW\include\ddk\ndis.h> ULONG MicrosecondsToSleep = 500; NdisMSleep(MicrosecondsToSleep);
//驱动的方法会出一堆错误

cap_result capture_result;
image im;
int cut_width = 416; int cut_height = 416; int channels = 3;

double frame_time = 1.0 / 80 * 1000; 
double control_time = 1.0 / 80 * 1000;  //ms 0.5是补偿，让总体时间刚好控制在1.0 / 80 * 1000, 12.5ms左右
float image_cut_data[519168]; //416*416*3
//image_cut_data = (float*)malloc(cut_width * cut_height * channels);

//HANDLE WaitTimer;

void capture()
{
    //HANDLE WaitTimer = CreateWaitableTimer(NULL, 1, NULL); 用于timer_microSleep(1, WaitTimer);

    /////调用截图cutImg();
    if (!Init())  //这里非常重要，初始化截图相关的东西，必须要有
    {
        Finit();
        printf("not support dxgi.\n");
    }
    int* pImgData1 = (int*)malloc(14745600); //用于下面的QueryFrame(pImgData1, imgLength);
    im.data = 0, im.w = cut_width, im.h = cut_height, im.c = channels;
    capture_result.im_float = 0, capture_result.im_unchar = 0, capture_result.new_flag = 0;

    cutImg();
    Sleep(10);
    cutImg();
    Sleep(10);
    cutImg();
    Sleep(10);
    capture_result = cutImg();
    im.data = capture_result.im_float;
    int image_cut_length = cut_width * cut_height* channels;
    memcpy(image_cut_data, im.data, image_cut_length);

    while (1)
    {
        if (GetAsyncKeyState(VK_LBUTTON)) //if (GetAsyncKeyState(VK_LBUTTON) && (!GetAsyncKeyState(0x10)))   
        {

            double time1 = get_time_chrono();
            capture_result = cutImg();
            im.data = capture_result.im_float;

            double cap_time = ((double)get_time_chrono() - time1) / 1000000;
            //printf("capture finished in %lf milli-seconds. frame time: %lf\n", cap_time, frame_time);

            if (cap_time < control_time)
            {
                int sleep_time = (int)((control_time - cap_time) * 1000);
                //printf("sleep_time%d", sleep_time);
                high_precision_microSleep(sleep_time); //微秒，这个精度可以控制在±3微秒,用于精确控制大于2ms的sleep，实际上并不是真正的sleep
                //microSleep(sleep_time); //这个的精度在1~2ms左右,是真的sleep，但是不够准确
                //nanoSleep(sleep_time*1000); //这个的精度一样是在1~2ms左右，要你何用
                //select_microSleep(500);
                //Sleep(1); //精度2ms.
                //SleepShort(sleep_time/1000);
            }
            printf("Total capture finished in %lf milli-seconds.\n", (((double)get_time_chrono() - time1) / 1000000));
            //printf("capture finished\n");  
        }
        else
        {
            //Sleep(0); CPU占用太高，就是一直循环睡眠2微秒。
            //select_microSleep(1); //精度900微秒，小于900的全都算900微秒。cpu占用为0
            //select_microSleep(500); //精度450微秒，小于450的全都算450微秒。select_microSleep(500)时会睡眠950微秒
            //microSleep(sleep_time); //这个的精度在1~2ms左右,是真的sleep，但是不够准确，cpu占用为0
            //nanoSleep(sleep_time*1000); //这个的精度一样是在1~2ms左右，要你何用
            //SleepShort(0.1); //SleepShort(0.1)，小于0.5睡0.45ms,精度450微秒；cpu占用为0

            double time1 = get_time_chrono();
            //CloseHandle(WaitTimer);
            //timer_microSleep(1, WaitTimer);
            //timer_microSleep_test1();
            
            ///int imgLength = 14745600;
            ///BOOL get_frame = QueryFrame(pImgData1, imgLength);
            ///printf("%d XXXXX in %lf milli-seconds.\n", get_frame, (((double)get_time_chrono() - time1) / 1000000));
            SleepShort(0.1); //SleepShort(0.5)睡0.9ms；SleepShort(0.1)，小于0.5睡0.45ms；SleepShort(1)睡1.45ms左右
            printf("XXXXX in %lf milli-seconds.\n", (((double)get_time_chrono() - time1) / 1000000));
        }
    }
}

int compare_image(float* tempdata, image img);
int compare_image(float* tempdata, image img)
{
    int i = 15000; //共查询10000个点，用来比较两个图是否一样
    int max = img.w * img.h * img.c;
    int min = 1;
    int pixel_n = 0;

    while (i--)
    {
        srand((unsigned)time(NULL)); /*用时间做种，每次产生随机数不一样*/
        pixel_n = rand(NULL) % (max - min + 1) + min; /*产生min~max的随机数*/
        if (tempdata[pixel_n] == im.data[pixel_n])
        {
            if (i <= 1) return 1;
            continue;
        }
        else
        {
            return 0;
            break;
        }
    }
    return 1;
}

void save_capture()
{
    //capture_result.im_unchar = 0;
    Sleep(3000); //让截图程序先初始化

    SYSTEMTIME currentTime;
    GetSystemTime(&currentTime);
    printf("time: %u/%u/%u %u:%u:%u:%u week-%d\n",
        currentTime.wYear, currentTime.wMonth, currentTime.wDay,
        currentTime.wHour, currentTime.wMinute, currentTime.wSecond,
        currentTime.wMilliseconds, currentTime.wDayOfWeek);

    char str_year[10], str_month[10], str_day[10], str_hour[10], str_min[10], str_sec[10], str_millsec[10];
    int number_year, number_month, number_day, number_hour, number_min, number_sec, number_millsec;
    //char str_timeX[50] = {"1234567890"};    //char a[10] = {"1234567890"} 是定义字符型数组，即a[0]为"1",a[1]为"2"


    int image_cut_length = cut_width * cut_height* channels;

    while (1)
    {
        if (GetAsyncKeyState(VK_LBUTTON))
        {
            double time2 = get_time_chrono();

            //filename
            GetSystemTime(&currentTime);
            char str_time[50] = "G:/CVTemp/";  //重置名称，否则会叠加
            number_year = currentTime.wYear;
            number_month = currentTime.wMonth;
            number_day = currentTime.wDay;
            number_hour = currentTime.wHour;
            number_min = currentTime.wMinute;
            number_sec = currentTime.wSecond;
            number_millsec = currentTime.wMilliseconds;
            _itoa_s(number_year, str_year, 10, 10);
            _itoa_s(number_month, str_month, 10, 10);
            _itoa_s(number_day, str_day, 10, 10);
            _itoa_s(number_hour, str_hour, 10, 10);
            _itoa_s(number_min, str_min, 10, 10);
            _itoa_s(number_sec, str_sec, 10, 10);
            _itoa_s(number_millsec, str_millsec, 10, 10);
            //printf("integer = %d string = %s\n", number, str_time);
            strcat_s(str_time, 50, str_year);
            strcat_s(str_time, 50, str_month);
            strcat_s(str_time, 50, str_day);
            strcat_s(str_time, 50, str_hour);
            strcat_s(str_time, 50, str_min);
            strcat_s(str_time, 50, str_sec);
            strcat_s(str_time, 50, str_millsec);
            //strcat_s(str_time, 50, ".bmp");
            LPCSTR filename = str_time;


            int flag_same = compare_image(image_cut_data, im);
            
            if (flag_same)
            {
                select_microSleep(1); //900微秒
                //#double cap_time = ((double)get_time_chrono() - time2) / 1000000;
                //#printf("!!save cmopare same in %lf milli-seconds. flag_same: %d\n", cap_time, flag_same);
                continue;
            }
            else
            {
                //capture_result = cutImg();
                //#image im = load_image_capture(capture_result.im_float, cut_width, cut_height, channels);
                cap_save_image(capture_result.im_unchar, filename , BMP, 100); //保存为BMP速度快，JPG要慢5ms左右
                //#double cap_time = ((double)get_time_chrono() - time2) / 1000000;
                //#printf("XXXsave finished in %lf milli-seconds. flag_same: %d\n", cap_time, flag_same);
                /*
                if (cap_time < control_time)
                {
                    int sleep_time = (int)((control_time - cap_time) * 1000);
                    microSleep(sleep_time); //微秒
                }
                printf("!!!!total Save capture in %lf milli-seconds.\n", ((double)get_time_chrono() - time2) / 1000000);
                //printf("save_capture finished\n");
                */
                memcpy(image_cut_data, im.data, image_cut_length);
                printf("Total Save in %lf ms.\n", ((double)get_time_chrono() - time2) / 1000000);
            }
        }
        else
            select_microSleep(1);
    }
}



void apex_get_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    static int frame_id = 0;
    frame_id++;

    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    //printf("selected_detections_num: %d\n", selected_detections_num);

    float x1, y1, width, height, center_x, center_y, delta_x, delta_y, distence, temp_distence;
    float move_x, move_y;
    float sum_r_move_x, sum_r_move_y, temp_r_move_x, temp_r_move_y, r_x, r_y, v_x, v_y;
    float r_temp1_x, r_temp1_y, r_temp2_x, r_temp2_y, r_temp3_x, r_temp3_y;
    int r_move_x, r_move_y;

    float lower = 0.25, offset = 3.8, v_scale = 0.35, kkk = 0;
    temp_distence = cut_width * cut_height / 4;
    r_temp1_x = 0, r_temp1_y = 0;
    r_temp2_x = 0, r_temp2_y = 0;
    r_temp3_x = 0, r_temp3_y = 0;
    temp_r_move_x = 0;
    temp_r_move_y = 0;
    sum_r_move_x = 0;
    sum_r_move_y = 0;
    r_move_x = 0, r_move_y = 0;

    if (selected_detections_num)
    {

        int i;
        for (i = 0; i < selected_detections_num; ++i)
        {
            const int best_class = selected_detections[i].best_class;
            printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
            x1 = (selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w;
            y1 = (selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h;
            width = selected_detections[i].det.bbox.w*im.w;
            height = selected_detections[i].det.bbox.h*im.h;
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n", x1, y1, width, height);

            /*
            int j;
            for (j = 0; j < classes; ++j)
            {
                if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                    printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
                }
            }*/

            center_x = width / 2 + x1;
            center_y = height / 2 + y1;
            delta_x = center_x - cut_width / 2;
            delta_y = center_y - cut_height / 2;
            distence = delta_x * delta_x + delta_y * delta_y;
            if (distence <= temp_distence)
            {
                move_x = delta_x * lower;  // 实测，当目标中心距离屏幕中心x像素时，win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, 0, 0, 0)就需要移动x像素
                move_y = (delta_y - height / offset) * lower;  // (-height / offset * lower)
                temp_distence = distence;
            }
        }
        // real x, y
        sum_r_move_x = temp_r_move_x + r_move_x;
        sum_r_move_y = temp_r_move_y + r_move_y;
        r_x = delta_x - sum_r_move_x;
        r_y = delta_y - sum_r_move_y;

        v_x = ((r_temp2_x - r_temp1_x) + (r_temp3_x - r_temp2_x)) / 2;  //# 用过去3张图片的位置推算目标的两次速度，计算平均值
        v_y = ((r_temp2_y - r_temp1_y) + (r_temp3_y - r_temp2_y)) / 2;
        if (GetAsyncKeyState(0x41)) //按下A
        {
            kkk = 8;
            //printf("AAAAAAAAAAAAAAAAA\n");
        }
        else if (GetAsyncKeyState(0x44))//按下D
            kkk = -8;
        else
            kkk = 0;
        v_x = v_x + kkk;
        r_move_x = (int)(move_x + v_x * v_scale);
        r_move_y = (int)(move_y + v_y * v_scale);

        mouse_event(MOUSEEVENTF_MOVE, r_move_x, r_move_y, 0, 0); //这里先用4/5吧，感觉有点用，不要一次到位，可以避免过调？
        //#printf("XXXXXXX %f %f %f %f %d %d\n", move_x, move_y, v_x, v_y, r_move_x, r_move_y);
        //# sleep(0.010)  # 在鼠标移动后，给屏幕时间刷新，保证获得的图是移动后的图
        
        //# 处理中间值
        r_temp3_x = r_x, r_temp3_y = r_y;
        r_temp1_x = r_temp2_x, r_temp1_y = r_temp2_y;
        r_temp2_x = r_temp3_x, r_temp2_y = r_temp3_y;
        temp_r_move_x = r_move_x;
        temp_r_move_y = r_move_y;
        //dt2 = time.perf_counter() - t2
        //print('total. (%.5fs)' % dt2)
    }
    else
    {
        //var = 1
        r_temp1_x = 0, r_temp1_y = 0;
        r_temp2_x = 0, r_temp2_y = 0;
        r_temp3_x = 0, r_temp3_y = 0;
        temp_r_move_x = 0;
        temp_r_move_y = 0;
        sum_r_move_x = 0;
        sum_r_move_y = 0;
    }

/*
    // image output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i) {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;
 
    }*/
    free(selected_detections);
}

void apex_get_detections_improve(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    static int frame_id = 0;
    frame_id++;

    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    //printf("selected_detections_num: %d\n", selected_detections_num);

    float x0, y0, width, height, center_x, center_y, delta_x, delta_y,temp_delta_x,temp_delta_y, distence, temp_distence;
    float x, y;
    float move_x, move_y;
    float sum_r_move_x, sum_r_move_y, temp_r_move_x, temp_r_move_y, r_x, r_y, v_x, v_y;
    float r_temp1_x, r_temp1_y, r_temp2_x, r_temp2_y, r_temp3_x, r_temp3_y;
    int r_move_x, r_move_y;

    float lower = 0.25, offset = 3.8, v_scale = 0.35, v_x_camera = 20;
    temp_distence = cut_width * cut_height / 4;
    x3_x1 = 0; y3_y1 = 0;
    x2_x1 = 0; y2_y1 = 0;
    temp_r_move_x1 = 0, temp_r_move_y1 = 0;
    temp_r_move_x2 = 0, temp_r_move_y2 = 0;
    r_move_x = 0, r_move_y = 0;
    x0 = 0; y0 = 0;
    x1 = 0; y1 = 0;
    x2 = 0; y2 = 0;
    x3 = 0; y3 = 0;
    v_x = 0; v_y = 0;
    delta_time1 = 0;
    delta_time2 = 0;
    delta_time = 0;

    if (selected_detections_num)
    {

        for (int i = 0; i < selected_detections_num; ++i)
        {
            const int best_class = selected_detections[i].best_class;
            printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
            x0 = (selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w;
            y0 = (selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h;
            width = selected_detections[i].det.bbox.w*im.w;
            height = selected_detections[i].det.bbox.h*im.h;
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n", x0, y0, width, height);

            /*
            int j;
            for (j = 0; j < classes; ++j)
            {
                if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                    printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
                }
            }*/

            center_x = width / 2 + x0;
            center_y = height / 2 + y0;
            x = center_x - cut_width / 2;
            y = center_y - cut_height / 2;
            distence = x * x + y * y;
            if (distence <= temp_distence)
            {
                x3 = x;
                y3 = y;
                temp_distence = distence;
            }
        }
        // real x, y
        //sum_r_move_x = temp_r_move_x + r_move_x;
        //sum_r_move_y = temp_r_move_y + r_move_y;
        x3_x1 = x3 - temp_r_move_x2 - temp_r_move_x1;
        y3_y1 = y3 - temp_r_move_y2 - temp_r_move_y1;
        x2_x1 = x2 - temp_r_move_x1;
        y2_y1 = y2 - temp_r_move_y1;
        
        delta_time1 = (time_end2 - time_end1)/1000000; delta_time2 = (time_end3 - time_end2)/1000000;
        
        if ((delta_time1 == 0) || (delta_time2 == 0))
        {
            v_x = 0;
            v_y = 0;
        }
        else
        {
            v_x = ((x2_x1 - x1) / delta_time1 + (x3_x1 - x2_x1) / delta_time2) / 2;  //# 用过去3张图片的位置推算目标的两次速度，计算平均值
            v_y = ((y2_y1 - y1) / delta_time1 + (y3_y1 - y2_y1) / delta_time2) / 2;
        }

        if (GetAsyncKeyState(0x41)) //按下A
        {
            v_x = v_x + v_x_camera;
            v_y = v_y + 0;
            //printf("AAAAAAAAAAAAAAAAA\n");
        }
        else if (GetAsyncKeyState(0x44))//按下D
        {
            v_x = v_x - v_x_camera;
            v_y = v_y + 0;
        }
        else
        {
            v_x;
            v_y;
        }
        //v_x = v_x + kkk;
        //v_y = v_y + 0;
        move_x = x3 + v_x * delta_time;  // 实测，当目标中心距离屏幕中心x像素时，win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, 0, 0, 0)就需要移动x像素
        move_y = (y3 - height / offset) + v_y * delta_time;  // (-height / offset * lower)
        r_move_x = (int)(move_x * lower);
        r_move_y = (int)(move_y * lower);

        mouse_event(MOUSEEVENTF_MOVE, r_move_x, r_move_y, 0, 0); //这里先用4/5吧，感觉有点用，不要一次到位，可以避免过调？
        //#printf("XXXXXXX %f %f %f %f %d %d\n", move_x, move_y, v_x, v_y, r_move_x, r_move_y);
        //# sleep(0.010)  # 在鼠标移动后，给屏幕时间刷新，保证获得的图是移动后的图

        //# 处理中间值
        //r_temp3_x = r_x, r_temp3_y = r_y;
        //r_temp1_x = r_temp2_x, r_temp1_y = r_temp2_y;
        //r_temp2_x = r_x, r_temp2_y = r_y;
        temp_r_move_x1 = temp_r_move_x2; temp_r_move_y1 = temp_r_move_y2;
        temp_r_move_x2 = r_move_x; temp_r_move_y2 = r_move_y;
        x1 = x2; y1 = y2;
        x2 = x3; y2 = y3;
        //dt2 = time.perf_counter() - t2
        //print('total. (%.5fs)' % dt2)
    }
    else
    {
        x3_x1 = 0; y3_y1 = 0;
        x2_x1 = 0; y2_y1 = 0;
        temp_r_move_x1 = 0, temp_r_move_y1 = 0;
        temp_r_move_x2 = 0, temp_r_move_y2 = 0;
        r_move_x = 0, r_move_y = 0;
        x0 = 0; y0 = 0;
        x1 = 0; y1 = 0;
        x2 = 0; y2 = 0;
        x3 = 0; y3 = 0;
        v_x = 0; v_y = 0;
        delta_time1 = 0;
        delta_time2 = 0;
        delta_time = 0;
    }

    /*
        // image output
        qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
        for (i = 0; i < selected_detections_num; ++i) {
            int width = im.h * .006;
            if (width < 1)
                width = 1;

            //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
            int offset = selected_detections[i].best_class * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = selected_detections[i].det.bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left = (b.x - b.w / 2.)*im.w;
            int right = (b.x + b.w / 2.)*im.w;
            int top = (b.y - b.h / 2.)*im.h;
            int bot = (b.y + b.h / 2.)*im.h;

            if (left < 0) left = 0;
            if (right > im.w - 1) right = im.w - 1;
            if (top < 0) top = 0;
            if (bot > im.h - 1) bot = im.h - 1;

        }*/
    free(selected_detections);
}



void apex_detector_multi()
{
    
    
    //char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    //if (gpu_list) {

    gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;

    //int clear = find_arg(argc, argv, "-clear");
    char *datacfg = "#apex2019_1class.data"; //必须放在cfg文件夹下，程序里面会自动转换成cfg/***
    char *cfgfile = "#apex_2pred_1class_416_1anchor_yolov3-tiny.cfg"; //必须放在cfg文件夹下
    char *weights = "backup/#apex_2pred_1class_416_1anchor_yolov3-tiny_last.weights";
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = "dog.bmp";
    float thresh = 0.5;
    float hier_thresh = 0.5;
    int dont_show = 1;
    int ext_output = 1;
    int save_labels = 0;
    char *outfile = 0;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weights) {
        load_weights(&net, weights);
    }
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf(" Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222); //使用了一个固定的随机数种子
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    int image_cut_length = cut_width * cut_height* channels;
    float nms = .5;    // 0.4F
    while (1)
    {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);

        
        //im.c;  //用darknet预测时，只需要3通道

        //unsigned char *image_cut_data = (unsigned char*)malloc(cut_width * cut_height * channels);
        Sleep(1000);  //让截图程序先初始化
        printf("network initial complete!");
        while (1)
        {
            if (GetAsyncKeyState(VK_LBUTTON))
            {
                double time1 = get_time_chrono();
                int flag_same = compare_image(image_cut_data, im); //比较每次的图片是否一样
                
                if (flag_same) //if (flag_same) //capture_result.new_flag
                {
                    //double compare_time = ((double)get_time_chrono() - time1) / 1000000;
                    //printf("!!detect cmopare same in %lf milli-seconds. flag_same: %d\n", compare_time, flag_same);
                    //microSleep(100); //微秒
                    SleepShort(0.1);
                    continue;
                }
                else //图片和前一次不一样才进行识别
                {
                    //saveBmpFile("2xx.bmp", image_cut_data, imgLengthCut); //保存为bmp格式
                    //char str_time[50] = "1xx.bmp";
                    //LPCSTR filename1 = str_time;
                    //saveBmpFile(filename1, image_cut_data, imgLengthCut);

                    //printf("w: %d h:%d\n", net.w, net.h);
                    //#image im = load_image(input, 0, 0, net.c);
                    //#image sized = resize_image(im, net.w, net.h);
                    int letterbox = 0;
                    //image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
                    layer l = net.layers[net.n - 1];

                    //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
                    //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
                    //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float));

                    //#float *X = sized.data; 不需要缩放尺寸了，节约时间
                    float *X = im.data;

                    /////只有识别部分，前面的都是相关的准备程序
                    //time= what_time_is_it_now();
                    double time2 = get_time_chrono();
                    network_predict(net, X);
                    //network_predict_image(&net, im); letterbox = 1;
                    //#printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_chrono() - time2) / 1000000);
                    //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

                    int nboxes = 0;
                    detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
                    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

                    apex_get_detections(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
                    //#printf("!!Compare + Predicted + NMS + Move in %lf milli-seconds.\n", ((double)get_time_chrono() - time1) / 1000000);
                    //#//#draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
                    //save_image(im, "predictions");
                    //i++;
                    memcpy(image_cut_data, im.data, image_cut_length);
                    printf("------------------------------------------!!All Detection Done in %lf milli-seconds.\n", ((double)get_time_chrono() - time1) / 1000000);
                }
            }
            else
            {
                //double time3 = get_time_chrono();
                //Sleep(1); //单位是毫秒，sleep 1ms时，else系统占用最少，基本为0，不知道为啥。 设置为0.5ms则会有很高的占用18%左右，在else时。因为sleep的最小单位是1ms，无法读取小数。
                //Sleep(1) 是睡眠1ms，实际测试了几种测时间的函数，应该还是蛮准的
                SleepShort(0.1); //睡眠900微秒
                //printf("sleep %lf milli-seconds.\n", ((double)get_time_chrono() - time3) / 1000000); //好像这个get_time_point(), 实测最小时间分辨率是2ms。微秒级睡眠microSleep()，设置小于1000时，显示全都是1.9ms
            }
        }
        /*
        if (!dont_show) {
            show_image(im, "predictions");
        }

        if (outfile) {
            if (json_buf) {
                char *tmp = ", \n";
                fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
        }

        // pseudo labeling concept - fast.ai
        if (save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }
        */
        //darknet自带的释放内存
        //free_detections(dets, nboxes);
        //free_image(im);
        //free_image(sized);

        //我编辑的，释放用于存储图片的内存
        //free(image_cut_data); //这个释放有问题
        

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }

    if (outfile) {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}

double time_end = 0, time_end1 = 0, time_end2 = 0, time_end3 = 0;
void apex_detector_signle()
{

    //char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    //if (gpu_list) {

    gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;

    //int clear = find_arg(argc, argv, "-clear");
    char *datacfg = "#apex2019_1class.data"; //必须放在cfg文件夹下，程序里面会自动转换成cfg/***
    char *cfgfile = "#apex_2pred_1class_416_1anchor_yolov3-tiny.cfg"; //必须放在cfg文件夹下
    char *weights = "backup/#apex_2pred_1class_416_1anchor_yolov3-tiny_last.weights";
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = "dog.bmp";
    float thresh = 0.5;
    float hier_thresh = 0.5;
    int dont_show = 1;
    int ext_output = 1;
    int save_labels = 0;
    char *outfile = 0;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weights) {
        load_weights(&net, weights);
    }
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf(" Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222); //使用了一个固定的随机数种子
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    int image_cut_length = cut_width * cut_height* channels;
    float nms = .5;    // 0.4F
    while (1)
    {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);


        //im.c;  //用darknet预测时，只需要3通道

        //unsigned char *image_cut_data = (unsigned char*)malloc(cut_width * cut_height * channels);

        ////////////////////////////////////////////////////////////////////////////调用截图cutImg();
        if (!Init())  //这里非常重要，初始化截图相关的东西，必须要有
        {
            Finit();
            printf("not support dxgi.\n");
        }
        //int* pImgData1 = (int*)malloc(14745600); //用于下面的QueryFrame(pImgData1, imgLength);
        im.data = 0, im.w = cut_width, im.h = cut_height, im.c = channels;
        capture_result.im_float = 0, capture_result.im_unchar = 0, capture_result.new_flag = 0;

        cutImg();
        Sleep(10);
        cutImg();
        Sleep(10);
        cutImg();
        Sleep(10);
        capture_result = cutImg();
        im.data = capture_result.im_float;
        int image_cut_length = cut_width * cut_height* channels;
        memcpy(image_cut_data, im.data, image_cut_length);

        while (1)
        {
            if (GetAsyncKeyState(VK_LBUTTON)) //if (GetAsyncKeyState(VK_LBUTTON) && (!GetAsyncKeyState(0x10)))   
            {

             //double time1 = get_time_chrono();
            double time1 = get_time_chrono();//时间应该是从图像更新后的一瞬间
                capture_result = cutImg(); //判断图像更新后，会自动处理图像，需要
            
                im.data = capture_result.im_float;

                int letterbox = 0;
                //image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
                layer l = net.layers[net.n - 1];

                //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
                //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
                //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float));

                //#float *X = sized.data; 不需要缩放尺寸了，节约时间
                float *X = im.data;

                /////只有识别部分，前面的都是相关的准备程序
                //time= what_time_is_it_now();
                //double time2 = get_time_chrono();
                network_predict(net, X);
                //network_predict_image(&net, im); letterbox = 1;
                //#printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_chrono() - time2) / 1000000);
                //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

                int nboxes = 0;
                detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
                if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

                apex_get_detections(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
                //#printf("!!Compare + Predicted + NMS + Move in %lf milli-seconds.\n", ((double)get_time_chrono() - time1) / 1000000);
                //#//#draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
                //save_image(im, "predictions");
                //i++;
                ///memcpy(image_cut_data, im.data, image_cut_length); 用于image compare

             double cap_time = (get_time_chrono() - time1) / 1000000;
                printf("------------------------------------------!!All Detection Done in %lf milli-seconds.\n", cap_time);
                //printf("capture finished in %lf milli-seconds. frame time: %lf\n", cap_time, frame_time);

                if (cap_time < control_time)
                {
                    int sleep_time = (int)((control_time - cap_time) * 1000);
                    //printf("sleep_time%d", sleep_time);
                    high_precision_microSleep(sleep_time); //微秒，这个精度可以控制在±3微秒,用于精确控制大于2ms的sleep，实际上并不是真正的sleep
                    //microSleep(sleep_time); //这个的精度在1~2ms左右,是真的sleep，但是不够准确
                    //nanoSleep(sleep_time*1000); //这个的精度一样是在1~2ms左右，要你何用
                    //select_microSleep(500);
                    //Sleep(1); //精度2ms.
                    //SleepShort(sleep_time/1000);
                }
                time_end3 = (double)get_time_chrono();
                printf("Total time control in %lf milli-seconds.\n", ((time_end3 - time1) / 1000000));
                time_end1 = time_end2;
                time_end2 = time_end3;
            }
            else
            {
                //Sleep(0); CPU占用太高，就是一直循环睡眠2微秒。
                //select_microSleep(1); //精度900微秒，小于900的全都算900微秒。cpu占用为0
                //select_microSleep(500); //精度450微秒，小于450的全都算450微秒。select_microSleep(500)时会睡眠950微秒
                //microSleep(sleep_time); //这个的精度在1~2ms左右,是真的sleep，但是不够准确，cpu占用为0
                //nanoSleep(sleep_time*1000); //这个的精度一样是在1~2ms左右，要你何用
                //SleepShort(0.1); //SleepShort(0.1)，小于0.5睡0.45ms,精度450微秒；cpu占用为0

                double time1 = get_time_chrono();
                //CloseHandle(WaitTimer);
                //timer_microSleep(1, WaitTimer);
                //timer_microSleep_test1();

                ///int imgLength = 14745600;
                ///BOOL get_frame = QueryFrame(pImgData1, imgLength);
                ///printf("%d XXXXX in %lf milli-seconds.\n", get_frame, (((double)get_time_chrono() - time1) / 1000000));
                SleepShort(0.1); //SleepShort(0.5)睡0.9ms；SleepShort(0.1)，小于0.5睡0.45ms；SleepShort(1)睡1.45ms左右
                printf("XXXXX in %lf milli-seconds.\n", (((double)get_time_chrono() - time1) / 1000000));
            }
        }
        //unsigned char *image_cut_data = (unsigned char*)malloc(cut_width * cut_height * channels);
    }
}

/*
void run_detector()
{
    /*
    int dont_show = find_arg(argc, argv, "-dont_show");
    int show = find_arg(argc, argv, "-show");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    */
/*
    //char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    //if (gpu_list) {

    gpu = gpu_index;
    gpus = &gpu;
    ngpus = 1;
    
    //int clear = find_arg(argc, argv, "-clear");

    char *datacfg = "#apex2019_1class.data"; //必须放在cfg文件夹下，程序里面会自动转换成cfg/***
    char *cfg = "#apex_2pred_1class_416_1anchor_yolov3-tiny.cfg"; //必须放在cfg文件夹下
    char *weights = "backup/#apex_2pred_1class_416_1anchor_yolov3-tiny_last.weights";
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = "dog.bmp";
    float thresh = 0.25;
    float hier_thresh = 0.5;
    int dont_show = 1;
    int ext_output = 1;
    int save_labels = 0;
    char *outfile = 0;
    test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile);
}
*/


#include <pthread.h>

int main()
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    gpu_index = 0;

#ifndef GPU
    gpu_index = -1;
#else
    if (gpu_index >= 0) {
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }
#endif

    int way = 1;
    int save_bmp = 0;
    pthread_t t_capture;
    pthread_t t_apex_detector_multi;
    pthread_t t_apex_detector_signle;
    pthread_t t_save_capture;
    if (way == 3)
    {
        

        // 创建线程A
        //******************************//
        //多线程的时候，识别程序可能在更改数据的时候进行数据读取并识别，会造成程序卡死，需要截图完成后再进行别的动作。
        if (pthread_create(&t_capture, NULL, capture, NULL) == -1) {
            puts("fail to create pthread t_capture");
            exit(1);
        }

        Sleep(1000); //1秒
        if (pthread_create(&t_apex_detector_multi, NULL, apex_detector_multi, NULL) == -1) {
            puts("fail to create pthread t_apex_detector_multi");
            exit(1);
        }
    }
    else if (way == 1)
    {
        if (pthread_create(&t_apex_detector_signle, NULL, apex_detector_signle, NULL) == -1) {
            puts("fail to create pthread t_apex_detector_signle");
            exit(1);
        }
    }

    if (save_bmp)
    {
        if (pthread_create(&t_save_capture, NULL, save_capture, NULL) == -1)
        {
            puts("fail to create pthread t1");
            exit(1);
        }
    }
    
    getchar();
    return 0;
}
