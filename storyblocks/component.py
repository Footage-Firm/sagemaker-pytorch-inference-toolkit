from typing import Optional, Sequence


def frame_generator(urls: Sequence[str], content_class: str, batch_output_dir: str, batch_size: int,
                    temporary_image_dir: Optional[str] = None) -> Sequence[frame]:
    """take an iterable of urls and turn them into frames we can process

    this will take a list of images or video files and:

        1. if they are videos, convert them into single image frames which we save on s3. if they
           are images, pass them straight through.
        2. collect the resulting list of image urls into possibly multiple batches of up to
           `batch_size` image files
        3. return a data structure which contains details about the process-able images
            + could be a dataframe with columns `batch_idx`, `source_s3_url`, `frame_idx`,
              `s3_frame_path`
            + could be a list of csv or txt files saved on s3 where each line contains records like
              we defined in the above dataframe

    e.g. suppose the input is like

        >>> ['s3://path/to/video-0001',
        ...  's3://path/to/video-0002',
        ...  ...
        ...  's3://path/to/video-9999']

    our output could be a dataframe like

        | batch_idx | source_s3_url           | frame_idx | s3_frame_path                |
        |-----------|-------------------------|-----------|------------------------------|
        | 0         | s3://path/to/video-0001 | 0         | s3://path/to/frame-0001-0000 |
        | 0         | s3://path/to/video-0001 | 1         | s3://path/to/frame-0001-0001 |
        | 0         | s3://path/to/video-0001 | 2         | s3://path/to/frame-0001-0002 |
        | ...       | ...                     | ...       | ...                          |
        | 123       | s3://path/to/video-9999 | 17        | s3://path/to/frame-9999-0017 |
        | 123       | s3://path/to/video-9999 | 18        | s3://path/to/frame-9999-0018 |

    we probably need to make it such that single source files are not split across batches

    Args:
        urls: an iterable of urls containing content of type content_class
        content_class: one of `video` or `image` (audio if you have an idea what that means)
        batch_output_dir: s3 path where we will store all the batches we create
        batch_size: number of items that we want in a single output batch
        temporary_image_dir: if we need to generate new images (e.g. save frames from the videos we
            process), this is the directory where we do it

    """
    raise NotImplementedError()


def batch_scorer(frames: Sequence[frame], batch_transform_output_dir: str,
                 batch_size: int) -> Sequence[str]:
    """given an iterable of frames (whatever we choose as the output of frame_generator), collect
    them into batches, create csv files we can pass to sagemaker, and submit batch transform
    requests for those batches

    batch transforms in sagemaker take as their input files where each row is something to score.
    the outputs are files with the same base file names in a different directory
    (`batch_transform_output_dir`), and whose lines include the prediction output and can
    also contain parts of the input record (so that you can know e.g. which source url and frame
    number you processed).

    this will probably also need to support arguments for the infra we want to launch the transform
    jobs on, like instance type, number, etc.

    return the batch transform output file paths.

    Args:
        frames: a dataframe or some other iterable of frame information -- should contain *at least*
            the s3 paths to be submitted in the batch transform, but
        batch_transform_output_dir: where sagemaker will save the output of a batch transform run
        batch_size: number of items that we want in a single output batch. doesn't have to be the
            same as the frame batch size, but I'm not sure if it will be necessary to make them
            distinct

    """
    raise NotImplementedError()


def score_processor(s3_batch_transform_output_files: Sequence[str], clean_up_frames: bool = True):
    """take a list of s3 urls pointing to batch transform output files and process their contents

    "processing" will include several steps, e.g.

        1. parse s3 file into dataframe
        2. convert json output (e.g. frame bounding box locations and keypoints) into struct
        3. persist records to delta tables
        4. delete temporary images (don't keep temporary frames for frame generator, e.g.)

    Args:
        s3_batch_transform_output_files: iterable of files output by sagemaker batch transform
        clean_up_frames: whether or not we should delete the frame image after processing

    """
    raise NotImplementedError()