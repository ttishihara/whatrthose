var $form = $("#upload_form");
var $file = $("#file_selector");
var $uploadedImg = $("#uploaded_image");
var $helpText = $("#helpText");

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $uploadedImg.html('<img src="'+e.target.result+'"/>');
      $uploadedImg.show();
      $("#drag-files-label").html(input.files[0].name);
      //$uploadedImg.css('background-image', 'url('+e.target.result+')');
      //$uploadedImg.css('display', 'table-cell');
    };

    reader.readAsDataURL(input.files[0]);
  }
}

$file.on('change', function(){
  readURL(this);
  $form.addClass('loading');
});

$uploadedImg.on('webkitAnimationEnd MSAnimationEnd oAnimationEnd animationend', function(){
  $form.addClass('loaded');
});

$helpText.on('webkitAnimationEnd MSAnimationEnd oAnimationEnd animationend', function(){
  setTimeout(function() {
    $file.val('');
    $form.removeClass('loading').removeClass('loaded');
  }, 5000);
});

function setup_camera() {
    $('#my_camera').show();
    Webcam.reset();
    Webcam.attach('#my_camera');

    $('#start_camera').hide();  // Hide "Access Camera" button.
    $('#cancel_camera').show();  // Show "Cancel Camera" and "Snapshot" buttons.
    $('#snapshot').show();
    $('#retake_snapshot').hide();  // Hide "Retake Snapshot" button.
    $('#upload_form').replaceWith($('#upload_form').clone());  // Reset drag-and-drop form.
    $('#upload_form').hide();  // Hide Flask-wtf upload forms.
    $('#uploaded_image').hide();
}

function stop_camera() {
    $('#my_camera').hide();
    Webcam.reset();
    $('#cancel_camera').hide();
    $('#snapshot').hide();
    $('#retake_snapshot').hide();
    $('#start_camera').show();
    $('#upload_form').show();
    $('#my_camera').html("");
    $('#webcam_submit_button').hide();
}

function take_snapshot() {
    Webcam.snap( function(data_uri) {
        $('#my_camera').html('<img id="webcam_img" src="'+data_uri+'"/>');
    });
    $('#snapshot').hide();
    $('#retake_snapshot').show();
    $('#webcam_submit_button').show();
}

function webcam_submit() {
    var img = $('#webcam_img').attr('src');

    $.ajax({
        url: './webcam_submit',
        type: 'POST',
        data: $.param({file: img}),
        dataType: 'json',
        success: function(data, textStatus) {
            if (data.redirect) {
                window.location.href = data.redirect;
            }
            else { 
                window.location.href = data.results;
            }
        }
    });
}
