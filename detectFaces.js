const cv = require('opencv4nodejs');

async function detectFaces(imagePath) {
    try {
        if (!imagePath) {
            console.error('Error: Missing image path');
            return;
        }

        const img = await cv.imreadAsync(imagePath);
        const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
        const faces = await classifier.detectMultiScaleAsync(img);
        faces.forEach(faceRect => {
            const { x, y, width, height } = faceRect;
            img.drawRectangle(new cv.Point2(x, y), new cv.Point2(x + width, y + height), new cv.Vec3(0, 255, 0), 2);
        });

        cv.imshow('Detected Faces', img);
        cv.waitKey();
    } catch (error) {
        console.error('Error:', error);
    }
}

module.exports = detectFaces;
