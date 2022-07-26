#' Install system requirements for using YOLO v7 via Python
#'
#' @return `TRUE` (invisibly) on success
#'
#' @export
ovml_yolo7_python_setup <- function() {
    ## 1. install python if needed
    pyexe <- reticulate::install_python()

    ## 2. create the yolov7 virtual environment if needed, or find the existing on
    envname <- "ovml-yolov7"
    if (!envname %in% reticulate::virtualenv_list()) {
        reticulate::virtualenv_create(envname, python = reticulate::install_python(), packages = c("opencv-python", "torch", "pandas", "torchvision", "tqdm", "matplotlib", "seaborn", "pyyaml"))
        ## or create without packages and then
        ## reticulate::virtualenv_install(envname, packages = c("opencv-python", "torch", "pandas", "torchvision", "tqdm", "matplotlib", "seaborn", "pyyaml"))
    }

    ## 3. install yolov7 if needed from https://github.com/WongKinYiu/yolov7
    y7dir <- ovml_yolo7_python_dir(install = TRUE)
    ## 4. copy our py file
    file.copy(system.file("extdata/yolov7/python/yolor.py", package = "ovmlpy", mustWork = TRUE), y7dir, overwrite = TRUE)

    invisible(TRUE)

}

ovml_yolo7_python_dir <- function(install = FALSE) {
    y7dir <- rappdirs::user_data_dir("ovml", appauthor = "openvolley")
    y7dir <- file.path(y7dir, "yolov7")
    if (!dir.exists(y7dir)) dir.create(y7dir, recursive = TRUE)
    det_exe <- dir(y7dir, recursive = TRUE, full.names = TRUE, pattern = "detect\\.py")
    if (length(det_exe) < 1 && install) {
        tf <- tempfile(pattern = ".zip")
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
    envname <- "ovml-yolov7"
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
    ##    imsz <- magick::image_info(magick::image_read(image_file))
    imsz <- list(width = 1280, height = 720) #!!!
    ry7 <- reticulate::import_from_path("yolor", path = ovml_yolo7_python_dir())
    if (!"ovml_detfun" %in% names(net)) net$ovml_detfun <- "detect"
    if (net$ovml_detfun %in% "detect_pose") {
        if (missing(as) || length(as) < 1 || !as %in% c("keypoints", "segments")) as <- "segments"
        pose <- ry7$detect_pose(net, source = image_file, conf_thres = conf, iou_thres = nms_conf)
        process_pose_dets(pose, original_w = imsz$width, original_h = imsz$height, input_image_size = net$imgsz, as = as, letterboxing = FALSE)
    } else {
        ## the python detection handles either a single file name, directory name, file glob pattern
        ## if we've been given more than one image_file input, loop over them
        out <- do.call(rbind, lapply(seq_along(image_file), function(i) {
            blah <- reticulate::py_capture_output(reticulate::py_suppress_warnings(det <- ry7$detect(net, source = image_file[i], conf_thres = conf, iou_thres = nms_conf, classes = classes)))
            imgs <- det[[2]] ## file names, in case we passed e.g. a directory name
            det <- det[[1]]
            ## dets are class xywh conf (xywh normalized)
            data.frame(outer_i = i, image_number = as.integer(det[, 1]), class = net$names[det[, 2] + 1L], score = det[, 7], xmin = round((det[, 3] - det[, 5] / 2) * imsz$width), xmax = round((det[, 3] + det[, 5] / 2) * imsz$width), ymax = round((1 - det[, 4] + det[, 6] / 2) * imsz$height), ymin = round((1 - det[, 4] - det[, 6] / 2) * imsz$height), image_file = imgs[det[, 1]])
        }))
        ## re-count image numbers
        out$image_number <- as.integer(c(0, cumsum(diff(out$outer_i) | diff(out$image_number) != 0))) + 1L
        out[, setdiff(names(out), "outer_i")]
    }
}
