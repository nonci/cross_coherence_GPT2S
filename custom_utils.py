import torch #, time
from typing import List
#from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler
import random, os
import matplotlib.pyplot as plt


def tensor_rotate(t: torch.Tensor):
	return torch.cat([t[1:], t[0:1]])
	
def list_rotate(l: list):
	return l[1:] + l[0:1]

class RandomBatchOrderSampler(Sampler[List[int]]):
	'''
	Read a batch from a pre-batched file. Shuffling is performed inside batch only (using shuffle_every=batch_size).
	adapted from: https://glaringlee.github.io/_modules/torch/utils/data/sampler.html
	'''
	# TODO: impl. drop_last=False
	
	def __reset__(self):
		self.batch_indexes = list(range(len(self)))
		#self._curr = 0
		if self.shuffle:
			random.shuffle(self.batch_indexes)
			#print(self.batch_indexes[:6])
	
	def __init__(self, N: int, batch_size: int, drop_last: bool, shuffle_batch_indexes: bool = True) -> None:
		# Since collections.abc.Iterable does not check for `__getitem__`, which
		# is one way for an object to be an iterable, we don't do an `isinstance`
		# check here.
		if isinstance(batch_size, bool) or  batch_size <= 0:  #not isinstance(batch_size, _int_classes)
			raise ValueError("batch_size should be a positive integer value, "
							 "but got batch_size={}".format(batch_size))
		self.sampler = None
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.N = N
		self.shuffle = shuffle_batch_indexes
		self.batch_indexes = []  # init by __reset__
		print('[DEBUG]  Initializing custom sampler, N:', self.N, 'bsize:', self.batch_size)
		#self._curr = 0

	def __next__(self):
		# for _ in range( self.N // self.batch_size ):
		if len(self.batch_indexes)>0:
			idx_curr = self.batch_indexes.pop()
			batch = list(range(idx_curr*self.batch_size, idx_curr*(self.batch_size)+self.batch_size))
			return batch  # print(f'yielding batch {idx_curr}')  - yield
		else:
			raise StopIteration

	def __iter__(self):
		if not self.drop_last:
			raise NotImplementedError
		if len(self.batch_indexes)==0:
			self.__reset__()  # re-generate indexes list (+reshuffle)
			#print('RESET')
		return self


	def __len__(self):
		if self.drop_last:
			return self.N // self.batch_size  # type: ignore
		else:
			raise NotImplementedError
			#return (self.N + self.batch_size - 1) // self.batch_size  # type: ignore
'''
r = iter(RandomBatchOrderSampler(26, 1, True))
for b in range(6):
	print(next(r))
	
for b in range(27):
	print(next(r))
'''


def permutation_indexes(batch_size, device='cpu'):
	''' Return:
	batch_size**2 permutation indexes that allow to permute modulo batch_size
	batch_size "col.labels": new labels per each col. of a batch_size x batch_size matrix (same as perm.indexes[:batch_size])
 	batch_size "row labels": new labels per each row. of a batch_size x batch_size matrix
 	'''
	
	perm, new_row_labels = [], []
	p = torch.randperm(batch_size).to(device) #torch.tensor(range(batch_size))
	
	for i in range(batch_size):    
		perm.append(p+batch_size*i)
		new_row_labels.append((p==i).nonzero(as_tuple=True)[0])
	perm = torch.concat(perm).to(device)
	new_row_labels = torch.tensor(new_row_labels).to(device)  # CE with logits
	new_col_labels = perm[:batch_size]  # CE with logits.T
	
	return perm, new_col_labels, new_row_labels


def visualize_pointcloud_pair(pc1, pc2, txt='', filename=None, cc1=-1, cc2=-1, flip='xyz', marker=('o',4)):
	"""
	Visualize two point clouds using Matplotlib, RGB or not.

	Args:
		pc*: (torch.Tensor): torch Tensor with shape (N, 3) or (N, 6)
		filename (str, optional): Filename to save the figure. If set to None, plt.show will be called.
		cc<n>: cross-coherence between pc<n> and txt
	"""
	flip = [{'x':0, 'y':1, 'z':2}[_] for _ in flip]
	fig = plt.figure(figsize=(12,5))
	ax1 = fig.add_subplot(121, projection="3d")
	ax2 = fig.add_subplot(122, projection="3d")
	ax1.set_xticks([]); ax2.set_xticks([])
	ax1.set_yticks([]); ax2.set_yticks([])
	ax1.set_zticks([]); ax2.set_zticks([])
	if pc1.size(1) == 6: # RGB cloud
		ax1.scatter(pc1[:, flip[0]], pc1[:, flip[1]], pc1[:, flip[2]], c=pc1[:,3:])
		avg_color = torch.mean(pc1[:,3:]).item()
		ax1.set_facecolor([min(1, 1-avg_color+0.2)]*3)
		ax1.w_xaxis.pane.fill = False
		ax1.w_yaxis.pane.fill = False
		ax1.w_zaxis.pane.fill = False
	elif pc1.size(1) == 3: # no-color cloud
		ax1.scatter(pc1[:, flip[0]], pc1[:, flip[1]], pc1[:, flip[2]],)
	else:
		raise ValueError("Invalid shape for points. Expected (N, 3) or (N, 6).")
	if pc2.size(1) == 6: # RGB cloud
		ax2.scatter(pc2[:, flip[0]], pc2[:, flip[1]], pc2[:, flip[2]], c=pc2[:,3:], marker=marker[0], s=marker[1])
		avg_color = torch.mean(pc2[:,3:]).item()
		ax2.set_facecolor([min(1, 1-avg_color+0.2)]*3)
		ax2.w_xaxis.pane.fill = False
		ax2.w_yaxis.pane.fill = False
		ax2.w_zaxis.pane.fill = False
	elif pc2.size(1) == 3: # no-color cloud
		ax2.scatter(pc2[:, flip[0]], pc2[:, flip[1]], pc2[:, flip[2]])
	else:
		raise ValueError("Invalid shape for points. Expected (N, 3) or (N, 6).")
	ax1.set_xlabel("X")
	ax1.set_ylabel("Y")
	ax1.set_zlabel("Z")
	ax1.set_title(f'{txt}\nCC1={cc1:.3f} {"PREDICTED" if cc1>cc2 else ""} --- CC2={cc2:.3f} {"PREDICTED" if cc1<cc2 else ""}', size=8)
	#ax1.text(0, -.5, 0, f'', size=10, ha="center", transform=ax1.transAxes)
	#ax2.text(0, -.5, 0, s=f'CC={cc2:.3f}', size=10, ha="center", transform=ax2.transAxes)
	
	if filename is not None:
		#print(end='saving figure ...')
		plt.savefig(os.path.join('figures', filename))
		#print(' OK')
	else:
		plt.show()

	plt.close(fig)


def visualize_pointcloud(pc, save_fig=True, filename=None, marker=('o',4), flip='xyz', title='', verbose=True):
	"""
	Visualize a point cloud using Matplotlib, RGB or not.

	Args:
		pc: (torch.Tensor): torch Tensor with shape (N, 3) or (N, 6)
		save_figure (bool, optional): Whether to save the figure. Defaults to False.
		filename (str, optional): Filename to save the figure. Required if save_figure is True.
	"""    
	flip = [{'x':0, 'y':1, 'z':2}[_] for _ in flip]
	fig = plt.figure(dpi=800)
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title(title)
	if verbose: print('pc shape: ', pc.shape)
	if pc.size(1) == 6:
		# RGB cloud
		ax.scatter(pc[:, flip[0]], pc[:, flip[1]], pc[:, flip[2]], c=pc[:,3:], marker=marker[0], s=marker[1])
	elif pc.size(1) == 3:
		ax.scatter(pc[:, flip[0]], pc[:, flip[1]], pc[:, flip[2]], marker=marker[0], s=marker[1])
	else:
		raise ValueError("Invalid shape for points. Expected (N, 3) or (N, 6).")
	
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	max_ = torch.max(pc[:, :3])
	min_ = torch.min(pc[:, :3])
	ax.set_xlim([min_, max_])
	ax.set_ylim([min_, max_])
	ax.set_zlim([min_, max_])
	
	if save_fig:
		if filename is None:
			raise ValueError("Filename is required when save_figure is True.")
		if verbose: print('saving figure...')
		plt.savefig(filename)
	else:
		plt.show()
	plt.close(fig)

	
def plot_cloud_p3d(t, title='', w=1500, h=1000, mark_size=2.):
	from pytorch3d.structures import Pointclouds
	from pytorch3d.vis.plotly_vis import plot_scene
	pointcloud = Pointclouds(points=[t[:,:3]], features=[t[:,3:]])
	fig = plot_scene({ title[:80]+'\n'+title[80:]: {"object": pointcloud} },	pointcloud_marker_size = mark_size)
	fig.update_layout(
    	autosize=False,
    	width=w,
    	height=h,
	)
	fig.show()