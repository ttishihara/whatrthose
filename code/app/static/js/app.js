function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#uploaded_image").html('<img src="'+e.target.result+'"/>');
      $("#uploaded_image").show();
      $("#drag-files-label").html(input.files[0].name);
      //$("#uploaded_image").css('background-image', 'url('+e.target.result+')');
      //$("#uploaded_image").css('display', 'table-cell');
    }

    reader.readAsDataURL(input.files[0]);
  }
}

$('#file_selector').on('change', function(){
  console.log("HERE");
  readURL(this);
  $('#upload_form').addClass('loading');
});

$('#uploaded_image').on('webkitAnimationEnd MSAnimationEnd oAnimationEnd animationend', function(){
  $("#upload_form").addClass('loaded');
});

$("helpText").on('webkitAnimationEnd MSAnimationEnd oAnimationEnd animationend', function(){
  setTimeout(function() {
    $('#file_selector').val('');
    $('#upload_form').removeClass('loading').removeClass('loaded');
  }, 5000);
});

function setup_camera() {
    $('#my_camera').show();
    //Webcam.reset();
	Webcam.set({
		width: 1280,
		height: 720,
		image_format: 'jpeg',
		jpeg_quality: 90,
		constraints: {
			width: { exact: 1280 },
			height: { exact: 720 }
		}
	});
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
    $('#uploaded_image').show();
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


// Google analytics event tracking
$('#webcam_submit_button').click(function() {
    console.log("HMM");
    gtag('event', 'submit', {'event_category': 'submission',
        'event_label': 'webcam'});
});
$('#submit').click(function() {
    gtag('event', 'submit', {'event_category': 'submission',
        'event_label': 'upload'});
});
$('#flightclub').click(function() {
    gtag('event', 'click', {'event_category': 'shop',
        'event_label': 'flightclub'});
});
$('#stockx').click(function() {
    gtag('event', 'click', {'event_category': 'shop',
        'event_label': 'stockx'});
});
