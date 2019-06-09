let base64Image;
$('#image-selector').change(function () {
    let reader = new FileReader();

    reader.onload = function(e) {
        let dataURL = reader.result;
        $('#selected-image').attr('src', dataURL);
        $('#predicted-image').attr('src', '');
        dataURLnoPNG = dataURL.replace('data:image/png;base64,', '');
        dataURLFinal = dataURLnoPNG.replace('data:image/jpeg;base64,', '');
        base64Image = dataURLFinal
    }

    reader.readAsDataURL($('#image-selector')[0].files[0]);
    $('#predict-pretext').empty();
    $('#predict-button').empty();

    $('#predicted-dog-label').empty();
    $('#predicted-dog-label').append('Predicted Dog');

    $('#predict-pretext').append('Now...predict the breed!');
    $('#predict-button').append('<button type="button" class="btn btn-outline-leaves">Predict</button>');

    $('#main-prediction').empty();
    $('#other-predictions').empty();
});

$('#predict-button').click(function(event) {
    $('html, body').animate({scrollTop:$(document).height()}, 'slow');

    $('#main-prediction').append('Predicting the image...');
    
    let message = {
        image: base64Image
    }

    $.post('$PORT/predict', JSON.stringify(message), function(response) {
        $('#main-prediction').empty();
        $('#other-predictions').empty();

        if (['a', 'e', 'i', 'o', 'u'].indexOf(response.prediction.class[0].charAt(0).toLowerCase()) !== -1) {
            $('#main-prediction').append('Is that an...' + response.prediction.class[0] + '?<br><span style="color: #c5001a;">' + response.prediction.percentage[0] + '</span> Match!');
        } else {
            $('#main-prediction').append('Is that a...' + response.prediction.class[0] + '?<br><span style="color: #c5001a;">' + response.prediction.percentage[0] + '</span> Match!');
        }

        if (parseFloat(response.prediction.percentage[0].slice(0, -1)) > 33) {
            $('#main-prediction').empty();
            if (['a', 'e', 'i', 'o', 'u'].indexOf(response.prediction.class[0].charAt(0).toLowerCase()) !== -1) {
                $('#main-prediction').append('Is that an...' + response.prediction.class[0] + '?<br><span style="color: #ffbb00;">' + response.prediction.percentage[0] + '</span> Match!');
            } else {
                $('#main-prediction').append('Is that a...' + response.prediction.class[0] + '?<br><span style="color: #ffbb00;">' + response.prediction.percentage[0] + '</span> Match!');
            }
        }

        if (parseFloat(response.prediction.percentage[0].slice(0, -1)) > 66) {
            $('#main-prediction').empty();
            if (['a', 'e', 'i', 'o', 'u'].indexOf(response.prediction.class[0].charAt(0).toLowerCase()) !== -1) {
                $('#main-prediction').append('Is that an...' + response.prediction.class[0] + '?<br><span style="color: #28a745;">' + response.prediction.percentage[0] + '</span> Match!');
            } else {
                $('#main-prediction').append('Is that a...' + response.prediction.class[0] + '?<br><span style="color: #28a745;">' + response.prediction.percentage[0] + '</span> Match!');
            }
        }

        $('#predicted-dog-label').empty();
        $('#predicted-dog-label').append(response.prediction.class[0]);
        $('#predicted-image').attr('src', 'static/assets/dog_images/' + response.prediction.class[0] + '.jpg');

        $('#other-predictions').append('Do you think that\'s the wrong breed? Try again with a different image, maybe the next one will be right!<br><br><u>Other Predictions:</u> ' + '<br>');
        for (var i = 1; i < response.prediction.class.length; i++) {
            $('#other-predictions').append(response.prediction.class[i] + ': ' + response.prediction.percentage[i] + '<br>')
        }
        $('#other-predictions').append("");
    });
});

$('#credits-link').click(function(event) {
    $('html, body').animate({scrollTop:$(document).height()}, 'slow');

    $('#credits-section').empty();
    $('#credits-section').append(
    '<div class="col">\
        <div class="card backColorBark colorMarble">\
            <div class="card-body">\
                <h3 class="card-title">Credits:</h3>\
                <p class="card-text">\
                    Made by: Some random guy who thought that this was a good idea.<br>\
                    Model used for transfer learning: <a href="https://keras.io/applications/#mobilenet">MobileNet</a><br>\
                    Libraries used: <a href="https://keras.io/">Keras</a> and <a href="https://www.tensorflow.org/">TensorFlow</a><br>\
                    Dataset used to train model and images for sample predicted dogs: <a href="https://www.kaggle.com/jessicali9530/stanford-dogs-dataset">Stanford Dogs Dataset</a><br>\
                    Have feedback to give? Email me at 1705408a@student.tp.edu.sg<br>\
                    Hope you enjoyed this little web application!\
                </p><br>\
            </div>\
        </div>\
    </div>'
    );
});