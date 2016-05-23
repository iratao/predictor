(function() {
	const DEFAULT_IMGURL = '../img/default_img.png';

	var itemlist = {'items': []};

	function getProductBySeekID() {
		var seekID = $('#seekid').val();

		$.ajax({
            type: 'GET',
            url: '/api/product/' + seekID,
            success: addProduct,
            error: getProductError,
        }); 
	};

	function getNextPredict() {
		$.ajax({
            type: 'GET',
            url: '/home/api/v1.0/predict_example',
            success: showPredictResult,
            error: showPredictErrorResult,
        }); 
	};

	function showPredictResult(data) {
		$('#mytable tbody').empty();
		// $('#right_panel').append('<div class="row"><p>' + JSON.stringify(data) + '</p></div>')
		Object.keys(data).forEach(function(key) {
			if (key === 'image') {return;}
			$('#mytable tbody').append('<tr><td>' + data[key].prediction + '</td><td>' + data[key].trueclass + '</td></tr>' );
		});
		$('#houseimg').attr('src', data.image)
	};

	function showPredictErrorResult(xhr, status, error) {
		$('#right_panel').empty();
		$('#right_panel').append('<div class="row"><p>' + JSON.stringify(data) + '</p></div>')
	}

	function getRoomType() {
		$.ajax({
            type: 'POST',
            url: '/api/classify',
            contentType: 'application/json',
            data: JSON.stringify(itemlist),
            success: getRoomTypeSuccess,
            error: getRoomTypeError,
            dataType: 'json',
        }); 
	};

	function getRoomTypeSuccess(data) {
		console.dir('getRoomTypeSucess: ' + data);
		$('#right_panel').append('<div class="row"><p>' + JSON.stringify(data) + '</p></div>')
	};

	function getRoomTypeError(xhr, status, error) {
		console.log(error);
	};

	function addProduct(data) {
		if (!data || data.length == 0) {
			return;
		}
		var item = JSON.parse(data[0]).item;
		var id = item.id;
		var name = item.name;
		var image = (item.imagesResize && item.imagesResize.length > 0) ?  item.imagesResize[0] : DEFAULT_IMGURL;
		var contentType = getContentType(item);
		
		itemlist.items.push({id, name, image, contentType});
		
		var html = '<div class="col-sm-4"><img src="' + image + '" alt="' + name +'" class="img-rounded"></div>';
		if (itemlist.items.length % 3 === 1) {
			$('#contentlist').append('<div class="row"></div');
		}
		$('#contentlist div.row:last-child').append(html);
		getRoomType();
	};

	function getProductError(jqXHR, textStatus, errorThrown) {
		console.log(jqXHR);
		console.log(textStatus);
		console.log(errorThrown);
	}

	function getContentType(item) {
		var attributes = item.attributes;
		for (var i = 0; i < attributes.length; i++) {
			var attr = attributes[i];
			if (attr.name === 'ContentType') {
				return attr.values && attr.values.length > 0 ? attr.values[0].value : undefined;
			}
		}
	}

	window.onload = function() {
		$("#add_button").click(getProductBySeekID);
		$("#next_button").click(getNextPredict);
	}


})();