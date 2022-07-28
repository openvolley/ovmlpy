#' Install system requirements for using YOLO v7 via Python
#'
#' Python and its required packages are installed into a virtual environment. `ovml_yolo7_python_envname()` returns the name of the virtual environment used, and `ovml_yolo7_python_envpath()` its path on the file system.
#'
#' @return `TRUE` (invisibly) on success
#'
#' @export
ovml_yolo7_python_setup <- function() {
    ## 1. install python if needed
    pyexe <- reticulate::install_python()

    ## 2. create the yolov7 virtual environment if needed, or find the existing on
    envname <- ovml_yolo7_python_envname()
    if (!envname %in% reticulate::virtualenv_list()) {
        reticulate::virtualenv_create(envname, python = reticulate::install_python(), packages = c("opencv-python", "torch", "pandas", "torchvision", "tqdm", "matplotlib", "seaborn", "pyyaml"))
    }

    ## 3. install yolov7 if needed from https://github.com/WongKinYiu/yolov7
    y7dir <- ovml_yolo7_python_dir(install = TRUE)
    ## 4. copy our py file
    file.copy(system.file("extdata/yolov7/python/yolor.py", package = "ovmlpy", mustWork = TRUE), y7dir, overwrite = TRUE)

    invisible(TRUE)

}

#' @rdname ovml_yolo7_python_setup
#' @export
ovml_yolo7_python_envname <- function() "ovml-yolov7"

#' @rdname ovml_yolo7_python_setup
#' @export
ovml_yolo7_python_envpath <- function() file.path(reticulate::virtualenv_root(), ovml_yolo7_python_envname)

ovml_yolo7_python_dir <- function(install = FALSE) {
    y7dir <- rappdirs::user_data_dir("ovml", appauthor = "openvolley")
    y7dir <- normalizePath(file.path(y7dir, "yolov7"))
    if (!dir.exists(y7dir)) dir.create(y7dir, recursive = TRUE)
    det_exe <- dir(y7dir, recursive = TRUE, full.names = TRUE, pattern = "detect\\.py")
    if (length(det_exe) < 1 && install) {
        tf <- tempfile(fileext = ".zip")
        download.file("https://github.com/WongKinYiu/yolov7/archive/refs/heads/main.zip", destfile = tf)
        utils::unzip(tf, exdir = y7dir)
    }
    det_exe <- dir(y7dir, recursive = TRUE, full.names = TRUE, pattern = "detect\\.py")
    if (length(det_exe) < 1) NULL else dirname(det_exe[1])
}

#' Construct YOLO network
#'
#' Models are implemented in Python and accessed via `reticulate`.
#'
#' @references https://github.com/WongKinYiu/yolov7
#' @param version integer or string: one of
#' - 7 or "7-tiny" : YOLO v7 or v7-tiny
#'
#' @param device string or numeric: "cpu" or 0, 1, 2 etc for GPU devices
#' @param weights_file string: either the path to the weights file that already exists on your system or "auto". If "auto", the weights file will be downloaded if necessary and stored in the directory given by [ovml_cache_dir()]
#' @param ... : currently ignored
#'
#' @return A YOLO network object
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#'
#' @export
ovml_yolo <- function(version = "7", device = "cpu", weights_file = "auto", ...) {
    if (is.null(ovml_yolo7_python_dir())) stop("cannot find system dependencies, have you run ovml_yolo7_python_setup()?")
    if (is.numeric(version)) version <- as.character(version)
    if (is.numeric(device)) device <- as.character(device)
    assert_that(version %in% c("7", "7-tiny", "7-mvb", "7-tiny-mvb", "7-w6-pose"))
    image_size <- 640L
    ## sort out the weights file
    detfun <- "detect"
    if (version == "7") {
        w_url <- "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        expected_sha1 <- "723b07225efa90d86eb983713b66fd8be82dfb9f"
    } else if (version == "7-tiny") {
        w_url <- "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
        expected_sha1 <- "c42311ff54e2a962725d6cac3b66d4b1e04eda2d"
    } else if (version == "7-tiny-mvb") {
        w_url <- "https://github.com/openvolley/ovmlpy/releases/download/v0.1.0/yolov7-tiny-mvb.pt"
        expected_sha1 <- "cbbd5f7b23d482b431c800c5e578893a9e78aa03"
    } else if (version == "7-w6-pose") {
        w_url <- "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
        expected_sha1 <- "9afe19a0cb2a48e9f60a354a676b9cade69d7e30"
        detfun <- "detect_pose"
        image_size <- 960L
    } else {
        ## "7-mvb"
        w_url <- "https://github.com/openvolley/ovmlpy/releases/download/v0.1.0/yolov7-mvb.pt"
        expected_sha1 <- "ee5f4f3a0c7cf06615c179805772ed5d6844e81c"
    }
    if (length(weights_file) && nzchar(weights_file) && !is.na(weights_file)) {
        if (identical(tolower(weights_file), "auto")) {
            weights_file <- ovml_download_if(w_url, expected_sha1 = expected_sha1)
        }
    }
    weights_file <- tryCatch(normalizePath(weights_file, mustWork = TRUE), error = function(e) NULL)
    if (is.null(weights_file) || !file.exists(weights_file)) stop("weights file does not exist")
    envname <- ovml_yolo7_python_envname()
    reticulate::use_virtualenv(envname)
    ## re-copy our py file in case this package has been updated
    file.copy(system.file("extdata/yolov7/python/yolor.py", package = "ovmlpy", mustWork = TRUE), ovml_yolo7_python_dir(), overwrite = TRUE)
    ry7 <- reticulate::import_from_path("yolor", path = ovml_yolo7_python_dir())
    blah <- reticulate::py_capture_output(out <- ry7$get_model(weights = weights_file, device = device, img_sz = image_size))
    out$ovml_detfun <- detfun
    out
}

#' Detect objects in image using a YOLO network
#'
#' Works on a single input image only, at the moment.
#'
#' @param net yolo: as returned by [ovml_yolo()]
#' @param image_file character: path to one or more image files, or a single video file (mp4, m4v, or mov extension)
#' @param conf scalar: confidence level
#' @param nms_conf scalar: non-max suppression confidence level
#' @param classes character: vector of class names, only detections of these classes will be returned
#' @param as string: for object detection networks, "boxes" (default and only option); for pose detection "segments" (default) or "keypoints"
#' @param ... : currently ignored
# @param batch_size integer: the number of images to process as a batch. Increasing `batch_size` will make processing of multiple images faster, but requires more memory
#'
#' @return A data.frame with columns "image_number", "image_file", "class", "score", "xmin", "xmax", "ymin", "ymax"
#'
#' @seealso [ovml_yolo()]
#'
#' @examples
#' \dontrun{
#'   dn <- ovml_yolo()
#'   img <- ovml_example_image()
#'   res <- ovml_yolo_detect(dn, img)
#'   ovml_ggplot(img, res)
#' }
#' @export
ovml_yolo_detect <- function(net, image_file, conf = 0.25, nms_conf = 0.45, classes, as, ...) {
    ##reticulate::use_virtualenv(envname)
    if (missing(classes)) {
        classes <- NULL
    } else {
        ## should be vector of class names to include
        ## convert to 0-based class number
        if (is.character(classes)) classes <- which(net$names %in% classes) - 1L
    }
    ry7 <- reticulate::import_from_path("yolor", path = ovml_yolo7_python_dir())
    if (!"ovml_detfun" %in% names(net)) net$ovml_detfun <- "detect"
    if (net$ovml_detfun %in% "detect_pose") {
        if (missing(as) || length(as) < 1 || !as %in% c("keypoints", "segments")) as <- "segments"
        out <- do.call(rbind, lapply(seq_along(image_file), function(i) {
            imsz <- magick::image_info(magick::image_read(image_file[i]))
            blah <- reticulate::py_capture_output(reticulate::py_suppress_warnings(pose <- ry7$detect_pose(net, source = image_file[i], conf_thres = conf, iou_thres = nms_conf)))
            do.call(rbind, lapply(pose, function(z) {
                this <- process_pose_dets(z[[3]], original_w = imsz$width, original_h = imsz$height, input_image_size = net$imgsz, as = as, letterboxing = FALSE)
                if (is.null(this)) {
                    this <- if (as == "segments") data.frame(outer_i = i, image_number = 1L, object = NA_integer_, segment = NA_integer_, x1 = NA_real_, x2 = NA_real_, y1 = NA_real_, y2 = NA_real_, conf1 = NA_real_, conf2 = NA_real_, image_file = NA_character_) else data.frame(outer_i = i, image_number = 1L, object = NA_integer_, keypoint = NA_integer_, x = NA_real_, y = NA_real_, conf = NA_real_, image_file = NA_character_)
                } else {
                    this$outer_i <- i
                    this$image_number <- z[[1]]
                    this$image_file <- z[[2]]
                }
                this
            }))
        }))
    } else {
        ## the python detection handles either a single file name, directory name, file glob pattern
        ## if we've been given more than one image_file input, loop over them
        out <- do.call(rbind, lapply(seq_along(image_file), function(i) {
            imsz <- tryCatch(if (grepl("\\.(mp4|avi|mov|webm|m4v)$", image_file[i], ignore.case = TRUE)) av::av_video_info(image_file[i])$video else magick::image_info(magick::image_read(image_file[i])), error = function(e) NULL)
            if (is.null(imsz) || nrow(imsz) > 1) stop("could not determine dimensions of file: ", image_file[i], "\n")
            blah <- reticulate::py_capture_output(reticulate::py_suppress_warnings(det <- ry7$detect(net, source = image_file[i], conf_thres = conf, iou_thres = nms_conf, classes = classes)))
            imgs <- det[[2]] ## file names, in case we passed e.g. a directory name
            det <- det[[1]]
            if (nrow(det) > 0) {
                ## dets are class xywh conf (xywh normalized)
                data.frame(outer_i = i, image_number = as.integer(det[, 1]), class = net$names[det[, 2] + 1L], score = det[, 7], xmin = round((det[, 3] - det[, 5] / 2) * imsz$width), xmax = round((det[, 3] + det[, 5] / 2) * imsz$width), ymax = round((1 - det[, 4] + det[, 6] / 2) * imsz$height), ymin = round((1 - det[, 4] - det[, 6] / 2) * imsz$height), image_file = imgs[det[, 1]])
            } else {
                ## placeholder row
                data.frame(outer_i = i, image_number = 1L, class = NA_integer_, score = NA_real_, xmin = NA_real_, xmax = NA_real_, ymax = NA_real_, ymin = NA_real_, image_file = NA_character_)
            }
        }))
    }
    ## re-count image numbers
    ## note that image numbers are not guaranteed to be correct if directory names have been passed in
    out$image_number <- as.integer(c(0, cumsum(diff(out$outer_i) | diff(out$image_number) != 0))) + 1L
    out[!is.na(out$image_file), setdiff(names(out), "outer_i")]
}
