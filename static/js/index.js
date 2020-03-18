const $form = $("#form");
$(".btn-send").on("click", evt => {
  $form.submit();
  $form[0].reset();
  return false;
});
