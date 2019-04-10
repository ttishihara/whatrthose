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
