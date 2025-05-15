//
// Created by ubuntu on 2/8/23.
//
#include "yolov8-seg.h"
#include <chrono>

namespace fs = ghc::filesystem;

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{ argv[1] };
    const fs::path    path{ argv[2] };

    std::vector<cv::String> imagePathList;
    bool                     isVideo{ false };

    assert(argc == 3);

    auto yolov8_seg = new YOLOv8_seg(engine_file_path);
    yolov8_seg->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension().string();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path.string());
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
            || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob((path / "*.jpg").string(), imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size = cv::Size{ 640, 640 };
    int      topk = 100;
    int      seg_h = 160;
    int      seg_w = 160;
    int      seg_channels = 32;
    float    score_thres = 0.25f;
    float    iou_thres = 0.65f;

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path.string());

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8_seg->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_seg->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_seg->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8_seg->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_seg->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_seg->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_seg->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8_seg->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }

    cv::destroyAllWindows();
    delete yolov8_seg;
    return 0;
}