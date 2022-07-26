context("overall network tests")

dn <- NULL

test_that("basic inference", {
    skip_on_ci()
    dn <- ovml_yolo(version = "7")
    img <- ovml_example_image()
    res <- ovml_yolo_detect(dn, img, conf = 0.6)
    expect_true(setequal(res$class, c("person")))
    res <- ovml_yolo_detect(dn, img, conf = 0.2)
    expect_true(setequal(res$class, c("person", "bench", "tennis racket")))
    res <- ovml_yolo_detect(dn, img, conf = 0.2, classes = "bench")
    expect_true(setequal(res$class, c("bench")))
})


test_that("cuda fails on cpu-only system", {
    expect_error(ovml_yolo(device = 0), "CUDA unavailable")
})
