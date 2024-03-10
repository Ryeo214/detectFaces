const cv = require('opencv4nodejs');
const fs = require('fs');

async function detectFaces(input) {
    try {
        let cap;
        if (typeof input === 'string') { 
            if (fs.existsSync(input)) { 
                if (input.endsWith('.mp4') || input.endsWith('.avi')) { 
                    cap = new cv.VideoCapture(input);
                } else { 
                    const img = await cv.imreadAsync(input);
                    cv.imshow('Input Image', img);
                    cv.waitKey();
                    return;
                }
            } else {
                console.error('Error: File not found');
                return;
            }
        } else if (input instanceof cv.Mat) { 
            cap = input;
        } else {
            console.error('Error: Invalid input');
            return;
        }

        const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

        while (true) {
            const frame = cap.read();
            if (frame.empty) {
                break;
            }

            const grayFrame = await frame.cvtColorAsync(cv.COLOR_BGR2GRAY);
            const faces = await classifier.detectMultiScaleAsync(grayFrame);

            faces.forEach(faceRect => {
                const { x, y, width, height } = faceRect;
                frame.drawRectangle(new cv.Point2(x, y), new cv.Point2(x + width, y + height), new cv.Vec3(0, 255, 0), 2);
            });

            cv.imshow('Detected Faces', frame);
            const key = cv.waitKey(10); 
            if (key === 27) { 
                break;
            }
        }

        cv.destroyAllWindows();
    } catch (error) {
        console.error('Error:', error);
    }
}

module.exports = detectFaces;
