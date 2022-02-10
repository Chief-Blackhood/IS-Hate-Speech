# Multi model hate speech detection
The aim of the project is to reduce Hate Speech on Video Streaming Platforms.
Majority of the work done on this domain involves directly reading the comment and predicting it to be Hate or not.

However, this approach lacks as it fails to identify cases in which Toxic remarks or comments are made in the context of the video. An example of this can be shown below:
> The following [video](https://www.bitchute.com/video/2XXGIFJmsPH7/) shows a women tennis player Angelina Dimova playing tennis. <br> A comment such as *"That's a man for sure LOL!!"* spreads hate against her by calling her a man.
<br> Current work fails to recognize this as hate however in the context of the video this comment is hateful. We plan to include such examples in our dataset to allow for **Context Based Multimodal Hate Speech Detection**.

To try to solve this, we plan to create a dataset consisting of both Synthetic and Original comments along with the Video, Hate Towards Whom, what metadata is required for detection which help provide context behind the comment and whom the hate is directed towards.

**This repository is used to store all code that is used in the project, barring the dataset itself.**
