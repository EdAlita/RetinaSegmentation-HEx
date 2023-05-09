<?php
$dirname = $_GET['link'];
$images = scandir($dirname);
$ignore = Array(".", "..");
echo "<h1>$dirname</h1>";


foreach($images as $curimg){
    if(!in_array($curimg, $ignore)) {
        echo $curimg;
        echo "<br/>";
        echo "<img width='400' src='$dirname/$curimg' /><br>\n";
    }
}
?>

